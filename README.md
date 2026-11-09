# Mesa for big-endian PowerPC (PPC BE) with AMD Polaris (GFX8)

This is **Mesa 26.2.6** with fixes that make the AMD drivers work on a **big-endian PowerPC** host:
a Power Mac G5 (PPC970MP) with a Radeon RX 550 (Polaris 11). Stock Mesa assumes a little-endian
CPU for AMD GPUs; on a big-endian machine the drivers misread command buffers, descriptors, shader
data and application buffers.

| Branch | Mesa | Drivers | Packages |
|---|---|---|---|
| [`g5-26`](../../tree/g5-26) (default) | **26.2.6** | radeonsi (OpenGL 4.6 / ES 3.2), RADV (Vulkan 1.3), Rusticl (OpenCL 3.1), VA-API | `26.2.6-1+rx550g5.3` |
| [`g5-24`](../../tree/g5-24) | **24.1.6** | radeonsi (OpenGL 4.6 / ES 3.2), RADV (Vulkan 1.3), VA-API, VDPAU (built, untested) | `24.1.6-1+rx550g5.1` |

**Mesa 24.1.6 is also available** on the [`g5-24`](../../tree/g5-24) branch (the first complete release of this port; it still builds VDPAU, which Mesa removed in 25.3; VDPAU is untested).

Almost all changes take effect only on a big-endian CPU with a GFX8 GPU (RX 460/470/480/550/560/570/580);
other GPUs and CPUs run stock code. The port also needs the patched kernel driver
([linux-rx550-be](https://github.com/parabellum-push/linux-rx550-be)).

- Ready-to-install Debian/MintPPC ppc64 packages, install notes, test results and known issues:
  [rx550-g5](https://github.com/parabellum-push/rx550-g5).
- Our commits sit on top of the Mesa release tarball (tag `v26`); see them with
  `git log v26..g5-26`. Release tags: `24.1.6-1+rx550g5.1`, `26.2.6-1+rx550g5.2`, `26.2.6-1+rx550g5.3`.

This is an unofficial port, not the Mesa project: upstream Mesa lives at
https://gitlab.freedesktop.org/mesa/mesa (see README.rst). Building: as for upstream Mesa
(`docs/meson.rst`); the cross-build setup we use for the packages is in
[rx550-g5/packaging](https://github.com/parabellum-push/rx550-g5/tree/main/packaging/mesa).
