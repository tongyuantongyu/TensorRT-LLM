#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Binary-patches the OpenMPI 5 runtime shipped in the base image to fix a
# singleton-spawn bug: a fixed 50-byte buffer in ompi_pmix_print_id()
# (ompi/runtime/ompi_rte.c, OPAL_PRINT_NAME_ARGS_MAX_SIZE) silently truncates
# the "singleton.<hostname>.<pid>.<rank>" identity string passed to the
# auto-spawned PRTE daemon via `--singleton <id>` whenever hostname+pid push
# the string past ~49 characters. PRTE then misparses the truncated string
# (splitting on the last '.'), registering the wrong rank/nspace, and the
# MPI_Comm_spawn a singleton process does under the hood (e.g. MpiPoolSession)
# fails with a generic MPI_ERR_UNKNOWN. Longer/Kubernetes-style hostnames hit
# this reliably; see the internal root-cause writeup for the full trace.
#
# Not yet fixed upstream (checked open-mpi/ompi main). This patches the
# compiled library directly so we have a fix ahead of the next OpenMPI
# release + HPC-X pickup cycle; drop this once upstream ships the real fix
# and the base image's HPC-X is updated past it.
#
# The fix: grow the ring buffer's usable capacity for this one call from 1
# slot (50 bytes) to 2 contiguous slots (102 bytes) by (a) raising the
# snprintf maxlen for the "%s.%u" call from 50 to 102 and (b) capping the
# ring's wraparound at 15 slots instead of 16, so writing 102 bytes from any
# reachable index never runs past the fixed 16*51=816-byte ring. Both changes
# are localized immediate-operand edits in already-compiled instructions --
# no relinking, no relocation, no other byte touched. Per-architecture
# instruction-level detail is above each architecture's EDITS below.
#
# Why growing into the *next* ring slot is safe here specifically (and is
# NOT a general property of the ring -- do not copy this reasoning to a
# different call site without re-checking it): ompi_pmix_print_id() has
# exactly one caller in the entire OMPI tree --
#   opal_argv_append_nosize(&args, OMPI_PRINT_ID(OMPI_PROC_MYID));  // dpm.c
# -- and opal_argv_append_nosize() (opal/util/argv.c) strdup()s the
# returned string as essentially its next action, with nothing in between
# that could re-enter ompi_pmix_print_id or a sibling that shares the same
# ring. So there is no other code path in the process that could claim the
# next slot (and silently overwrite the tail of our still-live string)
# before it gets copied out. If ompi_pmix_print_id() ever grows a second
# caller -- e.g. something that holds two of its return values alive at
# once, like a printf with two %s args each fed by this function -- the
# second call claiming the neighboring slot *would* corrupt the tail of the
# first, still-pending one. (This is exactly what the ring's 16 slots are
# normally for: letting several recently-returned pointers stay valid
# simultaneously. Shrinking that to "the next call always safely overwrites
# whatever's there" is only true because there is, today, no second caller
# to race against -- it is not a property of the ring itself.) That can
# only happen via an OpenMPI source change, and any such change produces a
# different compiled binary, which the sha256 gate below already refuses to
# patch without a human re-checking it -- so this constraint doesn't need
# its own runtime guard, just this note for whoever re-derives the offsets
# after a rebase.
#
# This is tied to the exact libmpi.so shipped by the base image's pinned
# HPC-X build (checked against its sha256), not derived dynamically -- a
# rebuilt/updated HPC-X will have a different binary and this script must
# fail rather than guess. If the checksum doesn't match, that almost always
# means HPC-X/OpenMPI was updated: check upstream open-mpi/ompi first (the
# bug may already be fixed there), and if not, re-derive the offsets/edits
# above against the new binary and update the tables below.

set -Eeo pipefail

TARGET_LIB="/opt/hpcx/ompi5/lib/libmpi.so.40.40.7"
ARCH="$(uname -m)"

case "${ARCH}" in
    x86_64)
        # HPC-X 2.50 / OpenMPI 5.0.10rc2. In ompi_pmix_print_id():
        #   cmp $0x10,%edx -> cmp $0xf,%edx   @ +0x37   ring wraparound cap: 16 -> 15
        #   83 fa 10          83 fa 0f
        #   mov $0x32,%esi -> mov $0x66,%esi  @ +0x92   snprintf maxlen: 50 -> 102
        #   be 32 00 00 00    be 66 00 00 00
        #   (the mov's other 3 immediate bytes are 0x00 both before and
        #   after -- 50 and 102 both fit in the low byte -- so only that
        #   byte actually changes)
        KNOWN_SHA256="8e40c81db0ed59b5eda995a058047f1258dcf18417d9e8c292091e1ba5d8be26"
        PATCHED_SHA256="0d8eef9bb75218d07309eb62e78c3881053f2022072ec6e0de9f1bb4c69e9616"
        # offset:old_hex:new_hex, one edit per line
        EDITS="
        0x924e9:10:0f
        0x92543:32:66
        "
        ;;
    aarch64)
        # HPC-X 2.50 / OpenMPI 5.0.10rc2. Same function, same two edits. On
        # this build the compiler happens to reuse a single `cmp`'s flags for
        # both branches of ompi_pmix_print_id() (an intervening CBZ, not a
        # second re-compare), unlike x86_64 where each branch gets its own
        # copy. That means this one edit unavoidably also caps the
        # `procid == NULL` branch's wraparound, which never needed capping
        # (it only ever writes a short fixed string, well within one slot) --
        # a side effect of the compiler's code layout, not something aimed
        # for, and harmless since that branch just skips a slot 1-in-15 times.
        # There's no second, independent `cmp` to leave alone here:
        #   cmp w1, #16 -> cmp w1, #15   @ +0x2c   ring wraparound cap: 16 -> 15
        #   3f 40 00 71    3f 3c 00 71
        #   mov x1, #50 -> mov x1, #102  @ +0x70   snprintf maxlen: 50 -> 102
        #   41 06 80 d2    c1 0c 80 d2
        KNOWN_SHA256="5ede1ea30f2f5961f76368904812c2a88f2aa2d5033d78901b4f0b2bcafbc983"
        PATCHED_SHA256="51a4dda6b4e008b1ffaacd66616f4ef7752c63cde52fd17469d80d791b0309cc"
        EDITS="
        0x8af0d:40:3c
        0x8af50:4106:c10c
        "
        ;;
    *)
        echo "install_openmpi_singleton_patch.sh: unsupported architecture ${ARCH}" >&2
        exit 1
        ;;
esac

if [ ! -e "${TARGET_LIB}" ]; then
    echo "install_openmpi_singleton_patch.sh: ${TARGET_LIB} not found; base image layout changed, update this script" >&2
    exit 1
fi

actual_sha256="$(sha256sum "${TARGET_LIB}" | awk '{print $1}')"

if [ "${actual_sha256}" = "${PATCHED_SHA256}" ]; then
    echo "install_openmpi_singleton_patch.sh: ${TARGET_LIB} already patched, skipping"
    exit 0
fi

if [ "${actual_sha256}" != "${KNOWN_SHA256}" ]; then
    echo "install_openmpi_singleton_patch.sh: ${TARGET_LIB} (${ARCH}) sha256 (${actual_sha256}) doesn't match the pinned HPC-X 2.50 build this patch was derived against (${KNOWN_SHA256})." >&2
    echo "The base image's OpenMPI was updated. Check whether upstream open-mpi/ompi already fixed OPAL_PRINT_NAME_ARGS_MAX_SIZE truncation in ompi_pmix_print_id() (ompi/runtime/ompi_rte.c); if not, re-disassemble ompi_pmix_print_id and update the offsets/checksums in this script." >&2
    exit 1
fi

echo "install_openmpi_singleton_patch.sh: patching ${TARGET_LIB} (${ARCH}, sha256 ${actual_sha256})"

python3 - "${TARGET_LIB}" <<PYEOF
import sys

lib = sys.argv[1]
edits = []
for line in """${EDITS}""".splitlines():
    line = line.strip()
    if not line:
        continue
    off_s, old_s, new_s = line.split(":")
    edits.append((int(off_s, 16), bytes.fromhex(old_s), bytes.fromhex(new_s)))

with open(lib, "r+b") as f:
    for off, old, new in edits:
        f.seek(off)
        cur = f.read(len(old))
        if cur != old:
            sys.exit(f"expected {old.hex()} at offset 0x{off:x}, found {cur.hex()}; refusing to patch")
        f.seek(off)
        f.write(new)
        print(f"  patched offset 0x{off:x}: {old.hex()} -> {new.hex()}")
PYEOF

echo "install_openmpi_singleton_patch.sh: patched sha256 is now $(sha256sum "${TARGET_LIB}" | awk '{print $1}')"
echo "install_openmpi_singleton_patch.sh: done"
