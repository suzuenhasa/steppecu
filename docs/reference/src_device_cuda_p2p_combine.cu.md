# `p2p_combine.cu` reference

## 1. Purpose

`src/device/cuda/p2p_combine.cu` is the CUDA implementation of the fast multi-GPU
combine step. After steppe splits an f2 computation across several GPUs, each GPU
holds a *partial* result — the slice of the full f2 tensor that GPU computed. This
file assembles those slices back into one whole tensor.

It is the fast path. Instead of copying every GPU's slice down to host memory and
reassembling there, it keeps each slice where it already lives (in that GPU's
memory) and moves the bytes directly from GPU to GPU over the hardware's
peer-to-peer link. One GPU is chosen as the *root* (the assembly point). The root
copies its own slice into place, then pulls each other GPU's slice straight across
the link into place.

Because this file uses direct CUDA calls (`cudaMemcpyPeerAsync`, peer-access
enabling, streams), it is compiled only into the GPU layer of steppe. The rest of
the codebase — the core library, the public API, the command-line tool — never sees
a CUDA header. This file includes a CUDA-free declaration header so that the
CUDA-free entry point can call into here without itself pulling in any CUDA code.

The file provides two closely related functions and one private helper they share:

- `combine_f2_partials_resident` — assembles the whole tensor on the root, then
  copies it back to host memory once and returns it as a host object.
- `combine_f2_partials_resident_device` — assembles the whole tensor on the root
  and leaves it there, returning a GPU-resident handle with no copy back to host.
- `place_partials_into` — the private placement loop both of the above call.

### What changed and why it matters

An earlier version of this step copied every GPU's slice down to host memory, then
uploaded a reassembled tensor back to the GPU — a second full pass over the whole
tensor across the slow host link. On a large problem that second full copy back to
host cost about an extra second of wall-clock time. This file removes it entirely:
the per-GPU compute now leaves each slice resident on the GPU that produced it, and
this combine consumes those resident slices directly. There is no host round-trip,
no re-upload, no staging buffer, no zeroed accumulator, and no add step.

---

## 2. The two entry points

Both entry points do the same assembly and differ only in what they hand back.

| Function | Returns | Final copy to host? | Where the result lives |
|---|---|---|---|
| `combine_f2_partials_resident` | `F2BlockTensor` (host object) | Yes — one final copy of the whole tensor back to host | Host memory |
| `combine_f2_partials_resident_device` | `DeviceF2Blocks` (GPU handle) | No | Stays in the root GPU's memory |

Everything else is identical: the same up-front validation, the same binding to the
root GPU, the same fixed-order placement of every slice into its disjoint region,
and the same single wait for all the copies to finish. The *only* difference at the
placement call is which memory the slices are written into — a freshly allocated
pair of buffers that will be copied back to host, versus a pair of buffers inside
the GPU-resident result handle that is returned as-is.

The device-resident variant exists so a caller that will keep working on the GPU
does not pay for a copy back to host it does not need. The caller can ask for a copy
to host later, on demand. The assembled tensor is bit-for-bit identical between the
two functions — same bytes, same placement — because they run the same placement
loop; the only difference is whether it ends up in host memory or stays in GPU
memory.

### Shared inputs

Both functions take:

- `partials` — the per-GPU resident slices, in a fixed order (slice 0, slice 1,
  and so on). Each slice carries its own source GPU id and its own starting block
  offset, so the combine does not need a separate parallel list of GPU ids. An
  empty slice (a GPU that was assigned no work) has no resident buffers and places
  nothing.
- `shards` — the block-aligned plan describing which blocks each GPU owns. Kept so
  the validator can cross-check that the slices tile the whole tensor exactly.
- `P` — the population count (the side length of each square slab).
- `n_block_full` — the total number of blocks in the assembled tensor.
- `root_device_id` — which GPU is the assembly point and holds the full result.

Each function starts by calling a shared validator that fail-fast rejects a bad set
of inputs: wrong slice count, mismatched `P`, a slice that does not span exactly its
planned blocks, or slices that do not tile the whole block range. The host-memory
combine and this GPU combine use the *same* validator, so the two reject identical
bad inputs identically.

---

## 3. The shared placement loop

`place_partials_into` is the single copy of the placement logic that both entry
points call. It was factored out so that the two functions cannot drift apart: a fix
to how a peer slice is fetched lands in both at once. The two callers pass different
destination buffers but otherwise call it identically.

The tensor is laid out as `n_block_full` square slabs stacked one after another,
each slab being `P × P` doubles. The size of one slab (`P × P`) is called the
`slab`. A GPU that owns blocks starting at offset `b0` owns the contiguous region of
the tensor beginning at `slab × b0`.

The loop walks the slices in fixed order (slice 0, then 1, and so on) and for each
one:

1. **Places the per-block SNP counts.** Each slice carries the count of SNPs in each
   of its blocks. These are plain integers, copied into the output's block-count
   array at the slice's block offset. This is done on the host with no GPU math, and
   it is done even for an empty slice before skipping the rest.

2. **Skips empty slices.** A GPU that was assigned no work has no resident buffers,
   so there is nothing to copy — the loop moves on.

3. **Computes the destination region.** The slice's data occupies exactly
   `slab × (number of its blocks)` doubles, starting at byte offset `slab × b0`
   into the full result. Because the slices are block-aligned and disjoint, no two
   slices ever target the same region.

4. **Copies the bytes into place.** How depends on where the slice lives:
   - If the slice is on the root GPU itself, it is a plain GPU-to-GPU copy within
     the same GPU (`cudaMemcpyAsync` with device-to-device), moving it from its
     resident buffer into its region of the full result. No peer link is involved.
   - If the slice is on another GPU, the root enables peer access to that GPU (see
     section 4), then pulls it straight across the link with `cudaMemcpyPeerAsync`
     into its region — no trip through host memory, no staging buffer.

The copies are enqueued on the root's stream and are not waited on inside the loop;
a single wait happens once after the loop returns (see section 6).

### The peer-copy call signature

`cudaMemcpyPeerAsync` takes, in order: the destination pointer, the destination
GPU id, the source pointer, the source GPU id, the byte count, and the stream. The
destination GPU id is the root; the source GPU id is the slice's own GPU. This order
is easy to get subtly wrong, so it is worth stating plainly.

---

## 4. Peer access handling

Before the root can read another GPU's memory directly, peer access must be turned
on. There are three subtleties in how this file does it.

**Peer access is directional and is enabled from the reader's side.** The root is
the one doing the reading, so peer access is enabled while the root is the current
GPU, naming the *other* GPU as the peer to be reached. The call is
`cudaDeviceEnablePeerAccess(peer_gpu_id, 0)` issued with the root current.

**"Already enabled" is expected, not an error.** The very first combine turns the
link on; every combine after that finds it already on. CUDA reports this as a
specific "peer access already enabled" status. This file treats that status as a
normal, non-fatal outcome: the enable call goes through a non-throwing warn-style
check that tolerates it, rather than the throwing check used for real faults.

**The stale status is then cleared.** After a tolerated "already enabled", CUDA
leaves a sticky last-error flag set. If left alone, a later unrelated error check
could pick up this stale, already-handled status and misreport it. So immediately
after the enable, the code reads and discards the last error (`cudaGetLastError`
both reads *and* resets it), wiping the flag clean.

A *genuine* peer-enable failure — on a GPU the caller promised was reachable — is
not silently swallowed. It surfaces on the very next step: the actual peer copy runs
through the throwing check, so if the link truly cannot be established the copy
fails loudly. The caller is responsible for having verified up front that peer
access is permitted and the hardware supports it; by the time execution reaches this
file, that path has already been chosen, so this file does not re-probe whether peer
access is possible.

---

## 5. Why the result is bit-identical

This combine produces a result that is bit-for-bit identical to the host-memory
combine and to a single-GPU run. That guarantee is deliberate and rests on two
facts.

**The bytes are the same.** Each GPU's resident slice holds the exact f2 doubles its
matrix multiply produced. Because the work is split on block boundaries, each
block's bytes are identical to what a single-GPU run would have produced for that
block. The direct GPU-to-GPU copy and the peer copy only *move* those bytes — they
never recompute anything. The transport moves bytes; it does not do math.

**The placement is the same.** The slices are disjoint and together tile the whole
block range exactly (the validator confirms every block is covered exactly once), so
every slab of the result is written exactly once, by its owning GPU, at the same
offset the host-memory combine would use. A raw byte copy into that region is the
faithful placement.

Two things this file deliberately does *not* do, both for correctness:

- **No zeroing of the result first.** Because the tiling covers every slab exactly
  once, there is no gap to zero-fill and no overlap to accumulate. The result buffer
  is allocated but never cleared.
- **No add step.** Placement is a copy, never an addition onto an accumulator. An
  earlier design zeroed the result and then added each slice in. That add had a
  latent flaw: it could turn a negative-zero value into a positive zero, a
  difference that the earlier zero-fill happened to hide. A raw copy reproduces a
  negative-zero element exactly, just like the host-memory combine's element-wise
  copy. Removing the zero-fill-and-add both fixes that negative-zero flip and
  removes the extra pass.

Because both combine paths place the same bytes in the same fixed order, they are
interchangeable siblings that always agree, and both agree with a single-GPU run.
This is also why the combine is **never** an all-reduce style network reduction: an
all-reduce sums in an order that varies with the number of GPUs, which would break
the bit-for-bit guarantee. The fixed slice-by-slice placement is what keeps the
result reproducible.

---

## 6. Synchronization, lifetime, and device restore

### One wait, not one per slice

The placement loop enqueues every copy on the root's stream and does not wait inside
the loop. A single wait for the whole stream happens once, after the loop. There is
no per-slice synchronization.

This is safe because none of the source slices are freed while the loop runs. Each
slice lives on its handle until the caller frees it — which only happens *after* this
function returns. So there is nothing that could be freed out from under an
in-flight copy, and a single drain at the end covers every enqueued copy at once.
Waiting per slice would only add stalls between copies that are meant to overlap.

### Lifetime and ownership

This function reads the resident slices in place. It does not take ownership of them
and does not free them. The caller's collection of slices outlives the call, and the
caller frees the slices only after this function has returned. The final wait
guarantees every copy has completed before the function returns, so no copy is ever
still reading a slice that later gets freed.

### Binding to the root and restoring the caller's GPU

The whole routine runs with the root GPU current — the result buffer and every copy
target it. On entry the code records whichever GPU the caller had current, then binds
to the root. On exit it restores the caller's original GPU.

The restore is done with a scope-guard object whose destructor puts the original GPU
back. That makes the restore automatic and exception-safe: it fires on the normal
return path and also if the function throws partway through. The guard's restore
routes through a non-throwing check, so tearing down never throws. Restoring matters
because the caller drives each per-GPU compute by binding to that GPU in turn;
leaving the root bound would silently retarget a later call. Freeing the resident
slices later is unaffected by which GPU is current, because the allocator knows which
GPU each pointer belongs to.

### The combine's own stream

The combine creates its own non-blocking stream on the root rather than using the
per-GPU streams (which belong to the compute backends) or the legacy default stream.
Its own stream lets the peer copies pipeline and keeps the combine off the default
stream.

---

## 7. The pinned final copy back to host

This section applies only to `combine_f2_partials_resident`, the variant that
returns a host object.

After the single wait confirms the whole tensor is assembled on the root, the
function copies it back to host memory once. To make that copy fast, it temporarily
*pins* (page-locks) the host destination for the duration of the copy. Pinned host
memory transfers faster over the link. If pinning fails for any reason, the code
degrades gracefully to an ordinary transfer rather than failing.

The pinning is done with a scope-guard object that unregisters the memory when it
goes out of scope. The copy is enqueued and then waited on *before* that scope ends,
so the pin stays alive across the whole transfer and is only released after the copy
has finished.

Pinning is purely a speed choice and changes nothing about the result. Pinned and
ordinary transfers move the same bytes into the same placement, and this final copy
overwrites every element of the host tensor (the tiling covers the whole thing), so
the host object is never left with stale contents. The host tensor's storage is
sized but not pre-zeroed, since the copy overwrites all of it.

If the tensor is empty (zero total elements) the final copy is skipped entirely.
