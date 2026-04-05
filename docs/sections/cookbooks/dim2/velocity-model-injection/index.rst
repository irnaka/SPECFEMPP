.. _velocity_model_injection:

Injecting an External Velocity Model into GLL Points (2D)
=========================================================

This guide explains how to run a 2D SPECFEM++ simulation where the material
properties (P-wave velocity, S-wave velocity, and density) are taken from a
user-supplied external file instead of from the mesh database.  Every GLL
(Gauss-Lobatto-Legendre) quadrature point is independently assigned values
interpolated from the external grid, enabling spatially variable models at
sub-element resolution.

.. contents:: Table of contents
   :local:
   :depth: 2

Overview
--------

By default SPECFEM++ assigns a single, uniform set of material properties to
each spectral element, as defined in the mesh database (``database.bin``).
This feature lets you **replace** those properties at every GLL point by
supplying a velocity model file on a regular Cartesian grid or as a set of
scattered data points.

Typical use cases:

* Running a simulation on a mesh generated for a simple geometry (e.g.
  homogeneous half-space) but with a realistic velocity model from seismic
  tomography or a synthetic benchmark (Marmousi, SEAM, ...).
* Rapidly testing many velocity models on the same mesh without re-meshing.
* Injecting a model produced by an external inversion or forward-modelling
  code.

.. note::

   This feature currently supports **2D** simulations only.  The supported
   medium / property combinations are:

   * Elastic isotropic (PSV and SH)
   * Acoustic isotropic

   Poroelastic elements and anisotropic/Cosserat elastic elements are **not**
   overridden; they retain the material properties from the mesh database and
   a warning is printed.

Prerequisites
-------------

* SPECFEM++ compiled from source (see :ref:`getting_started`).
* A mesh database (``database.bin``) produced by ``xmeshfem2D``.
* A velocity model file covering the physical extent of the mesh.

No additional libraries are required; the velocity model reader is part of the
standard build.

Step 1 — Prepare your velocity model file
------------------------------------------

The reader supports three formats.  Choose whichever is most convenient.

.. _format_regular_ascii:

Format A: Regular-grid ASCII (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A regular Cartesian grid where ``x`` varies fastest (row-major, C order).

.. code-block:: text
   :caption: MODEL/velocity.dat  —  regular-grid ASCII format

   # REGULAR_GRID
   # NX  NZ   X0     Z0     DX      DZ
   201   101   0.0   0.0   100.0   100.0
   # Vp[m/s]  Vs[m/s]  Rho[kg/m^3]   (one row per grid point, x varies fastest)
   3000.0  1732.0  2700.0
   3000.0  1732.0  2700.0
   ...

Header layout:

* Line containing ``REGULAR_GRID`` (anywhere before the first data row,
  may appear inside a comment).
* One non-comment line: ``NX  NZ  X0  Z0  DX  DZ``

  * ``NX``, ``NZ`` — number of grid points in x and z directions
  * ``X0``, ``Z0`` — coordinates (metres) of the first grid point (bottom-left)
  * ``DX``, ``DZ`` — grid spacing (metres)

* ``NX × NZ`` data rows in the order: ``Vp  Vs  Rho``

  * Ordering: outer loop over z (slow), inner loop over x (fast).
  * The first ``NX`` rows correspond to ``z = Z0``, the next ``NX`` rows to
    ``z = Z0 + DZ``, etc.

.. important::

   ``X0`` and ``Z0`` must be ≤ the minimum x and z coordinates of any GLL
   point in the mesh, and ``X0 + (NX-1)*DX`` / ``Z0 + (NZ-1)*DZ`` must be ≥
   the maximum.  If a GLL point falls outside the grid, the behaviour is
   controlled by the ``out-of-bounds`` option (see :ref:`yaml_config`).

.. _format_scattered_ascii:

Format B: Scattered-point ASCII
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unstructured list of ``(x, z, Vp, Vs, Rho)`` tuples.  Nearest-neighbour
interpolation is used automatically.

.. code-block:: text
   :caption: MODEL/scattered.dat  —  scattered-point ASCII format

   # x[m]    z[m]    Vp[m/s]  Vs[m/s]  Rho[kg/m^3]
   0.0       0.0     3000.0   1732.0   2700.0
   500.0     0.0     3100.0   1790.0   2750.0
   1000.0    0.0     3200.0   1848.0   2800.0
   0.0       500.0   3050.0   1762.0   2720.0
   ...

There is no header line; comments (``#``) are ignored.  Every non-comment,
non-blank line must have exactly five whitespace-separated values.

.. note::

   For scattered data the reader always uses nearest-neighbour lookup
   regardless of the ``interpolation`` setting.  Bilinear interpolation
   requires a regular grid.

.. _format_binary:

Format C: Binary
~~~~~~~~~~~~~~~~

A compact binary file, useful for large models where ASCII I/O is slow.
All values are **little-endian**.

.. code-block:: text

   [ 4 bytes ] NX          (int32)
   [ 4 bytes ] NZ          (int32)
   [ 8 bytes ] X0          (float64)
   [ 8 bytes ] Z0          (float64)
   [ 8 bytes ] DX          (float64)
   [ 8 bytes ] DZ          (float64)
   [ NX*NZ × 8 bytes ] Vp  (float64, row-major, x varies fastest)
   [ NX*NZ × 8 bytes ] Vs  (float64)
   [ NX*NZ × 8 bytes ] Rho (float64)

The Python helper below can generate both ASCII and binary files.

Generating a model file with Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following self-contained script creates either format from a NumPy array.
Copy it to your working directory, edit the parameters at the top, and run it:

.. code-block:: python
   :caption: generate_model.py

   #!/usr/bin/env python3
   """
   Generate a SPECFEM++ GLL-injection velocity model file.

   Edit the parameters in the CONFIGURATION section and run:
       python generate_model.py
   """
   import numpy as np
   import struct

   # ── CONFIGURATION ───────────────────────────────────────────────────────────
   # Grid extent and spacing (must cover the full mesh domain)
   X0, Z0 = 0.0, 0.0          # bottom-left corner (metres)
   X1, Z1 = 20000.0, 10000.0  # top-right corner (metres)
   DX, DZ = 200.0, 100.0      # grid spacing (metres)

   # Output file paths
   ASCII_FILE  = "MODEL/velocity.dat"
   BINARY_FILE = "MODEL/velocity.bin"
   WRITE_ASCII  = True
   WRITE_BINARY = False
   # ────────────────────────────────────────────────────────────────────────────

   import os
   os.makedirs("MODEL", exist_ok=True)

   NX = int(round((X1 - X0) / DX)) + 1
   NZ = int(round((Z1 - Z0) / DZ)) + 1

   # ── BUILD YOUR MODEL HERE ────────────────────────────────────────────────────
   # Shape: (NZ, NX) — outer index is z (slow), inner index is x (fast)
   x_arr = X0 + np.arange(NX) * DX          # shape (NX,)
   z_arr = Z0 + np.arange(NZ) * DZ          # shape (NZ,)
   xx, zz = np.meshgrid(x_arr, z_arr)        # shape (NZ, NX)

   # Example: two-layer model separated at Z = 4000 m
   vp  = np.where(zz < 4000.0, 3000.0, 4500.0)  # m/s
   vs  = np.where(zz < 4000.0, 1732.0, 2598.0)  # m/s  (Vp/sqrt(3))
   rho = np.where(zz < 4000.0, 2700.0, 3000.0)  # kg/m^3
   # ────────────────────────────────────────────────────────────────────────────

   # Flatten row-major (x varies fastest)
   vp_flat  = vp.ravel()
   vs_flat  = vs.ravel()
   rho_flat = rho.ravel()

   if WRITE_ASCII:
       with open(ASCII_FILE, "w") as f:
           f.write("# REGULAR_GRID\n")
           f.write(f"# NX  NZ  X0  Z0  DX  DZ\n")
           f.write(f"{NX}  {NZ}  {X0}  {Z0}  {DX}  {DZ}\n")
           f.write("# Vp[m/s]  Vs[m/s]  Rho[kg/m^3]\n")
           for vp_v, vs_v, rho_v in zip(vp_flat, vs_flat, rho_flat):
               f.write(f"{vp_v:.4f}  {vs_v:.4f}  {rho_v:.4f}\n")
       print(f"Written {NX*NZ} points → {ASCII_FILE}")

   if WRITE_BINARY:
       with open(BINARY_FILE, "wb") as f:
           f.write(struct.pack("<ii", NX, NZ))
           f.write(struct.pack("<dddddd", X0, Z0, DX, DZ,
                               X0 + (NX-1)*DX, Z0 + (NZ-1)*DZ))
           # Rewrite with correct layout expected by SPECFEM++ binary reader:
           # header = nx, nz, x0, z0, dx, dz
           f.seek(0)
           f.write(struct.pack("<ii", NX, NZ))
           f.write(struct.pack("<dddd", X0, Z0, DX, DZ))
           vp_flat.astype("<f8").tofile(f)
           vs_flat.astype("<f8").tofile(f)
           rho_flat.astype("<f8").tofile(f)
       print(f"Written {NX*NZ} points → {BINARY_FILE}")

.. tip::

   For a **binary** file, use this simpler and correct version:

   .. code-block:: python

      with open(BINARY_FILE, "wb") as f:
          f.write(struct.pack("<ii", NX, NZ))
          f.write(struct.pack("<dddd", X0, Z0, DX, DZ))
          vp_flat.astype("<f8").tofile(f)
          vs_flat.astype("<f8").tofile(f)
          rho_flat.astype("<f8").tofile(f)

Step 2 — Configure the simulation
-----------------------------------

Add a ``velocity-model`` block inside the ``databases:`` section of your
``specfem_config.yaml``.

.. _yaml_config:

.. code-block:: yaml
   :caption: specfem_config.yaml  —  relevant excerpt

   parameters:

     # ... (header, simulation-setup, receivers, etc.) ...

     databases:
       mesh-database: "OUTPUT_FILES/database.bin"

       ## External velocity model injected at every GLL point.
       ## Remove or comment out this block to use mesh-database materials.
       velocity-model:
         file:          "MODEL/velocity.dat"   # path to model file (required)
         format:        ascii                   # "ascii" (default) or "binary"
         interpolation: bilinear                # "bilinear" (default) or "nearest"
         out-of-bounds: clamp                   # "clamp" (default) or "error"

     sources: "sources.yaml"

Parameter reference
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 12 12 56

   * - Parameter
     - Type
     - Default
     - Description
   * - ``file``
     - string
     - *(required)*
     - Path to the velocity model file.  Absolute or relative to the
       working directory from which ``specfem2d`` is run.
   * - ``format``
     - string
     - ``ascii``
     - File format.  ``ascii`` auto-detects regular-grid vs. scattered
       format from the presence of the ``REGULAR_GRID`` keyword.
       ``binary`` expects the compact binary layout described in
       :ref:`format_binary`.
   * - ``interpolation``
     - string
     - ``bilinear``
     - Interpolation method for regular grids.  ``bilinear`` performs
       bi-linear interpolation between the four surrounding grid nodes
       (recommended for smooth models).  ``nearest`` snaps to the
       closest grid node (faster; preserves sharp discontinuities).
       Scattered-point data always uses nearest-neighbour regardless of
       this setting.
   * - ``out-of-bounds``
     - string
     - ``clamp``
     - Behaviour when a GLL point falls outside the model grid.  ``clamp``
       silently extends the boundary value (safe default for models that
       almost cover the mesh).  ``error`` aborts with a descriptive
       message (useful to catch under-sized models during development).

.. note::

   ``velocity-model`` and the legacy ``databases.reader.properties`` key are
   mutually exclusive.  Defining both causes an error at startup.

Step 3 — Run the simulation
-----------------------------

Run the simulation exactly as you normally would:

.. code-block:: bash

   specfem2d -p specfem_config.yaml

SPECFEM++ will print a log message for each step of the injection:

.. code-block:: text

   [SF++][  INFO  ]: Reading/injecting GLL model from external reader.
   [SF++][  INFO  ]: Velocity model injection:
   [SF++][  INFO  ]:   File: MODEL/velocity.dat
   [SF++][  INFO  ]: CartesianGrid2D [regular grid]
   [SF++][  INFO  ]:   nx=201  nz=101
   [SF++][  INFO  ]:   x: [0, 20000]  dx=200
   [SF++][  INFO  ]:   z: [0, 10000]  dz=100
   [SF++][  INFO  ]:   total points: 20301
   [SF++][  INFO  ]: Velocity model injection complete.

After ``Velocity model injection complete.``, the simulation proceeds normally.

Complete worked example
-----------------------

This section walks through a full two-layer example from scratch using the
homogeneous-medium mesh (5 km × 5 km).

Directory layout
~~~~~~~~~~~~~~~~~

.. code-block:: text

   ~/specfempp-gll-example/
   ├── OUTPUT_FILES/          # mesh outputs
   ├── MODEL/
   │   └── velocity.dat       # generated by prepare_model.py
   ├── Par_file               # mesher parameter file
   ├── specfem_config.yaml
   ├── sources.yaml
   ├── STATIONS
   └── prepare_model.py

1. Create the workspace
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   mkdir -p ~/specfempp-gll-example/{OUTPUT_FILES,MODEL}
   cd ~/specfempp-gll-example

2. Generate the mesh
~~~~~~~~~~~~~~~~~~~~~

Use the same ``Par_file`` as the homogeneous-medium cookbook (see
:ref:`homogeneous_example`) but set the domain size to 5000 m × 5000 m.
Then run:

.. code-block:: bash

   xmeshfem2D -p Par_file

This produces ``OUTPUT_FILES/database.bin``.

3. Generate the velocity model file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save the following as ``prepare_model.py`` and run it:

.. code-block:: python
   :caption: prepare_model.py

   #!/usr/bin/env python3
   """
   Two-layer velocity model covering a 5 km × 5 km domain.
   Layer boundary at z = 2500 m (centre of the domain).
   """
   import numpy as np

   # Grid parameters — cover the full 5 km × 5 km mesh with 100 m spacing
   X0, Z0, DX, DZ = 0.0, 0.0, 100.0, 100.0
   NX, NZ = 51, 51          # 51 points × 100 m spacing = 5000 m extent

   x = np.linspace(X0, X0 + (NX - 1) * DX, NX)
   z = np.linspace(Z0, Z0 + (NZ - 1) * DZ, NZ)
   xx, zz = np.meshgrid(x, z)  # shape (NZ, NX)

   # Two-layer model: slow layer above, fast layer below z = 2500 m
   vp  = np.where(zz < 2500.0, 3000.0, 4500.0)   # m/s
   vs  = np.where(zz < 2500.0, 1732.0, 2598.0)   # m/s
   rho = np.where(zz < 2500.0, 2700.0, 3000.0)   # kg/m^3

   with open("MODEL/velocity.dat", "w") as f:
       f.write("# REGULAR_GRID\n")
       f.write(f"{NX}  {NZ}  {X0}  {Z0}  {DX}  {DZ}\n")
       for vp_v, vs_v, rho_v in zip(vp.ravel(), vs.ravel(), rho.ravel()):
           f.write(f"{vp_v:.2f}  {vs_v:.2f}  {rho_v:.2f}\n")

   print(f"Generated MODEL/velocity.dat  ({NX}×{NZ} = {NX*NZ} points)")

.. code-block:: bash

   python prepare_model.py
   # → Generated MODEL/velocity.dat  (51×51 = 2601 points)

4. Write the configuration files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml
   :caption: specfem_config.yaml

   parameters:

     header:
       title: Two-layer velocity model injection example
       description: |
         Two-layer elastic model injected at GLL points via external file.
         Layer boundary at z = 2500 m.

     simulation-setup:
       quadrature:
         quadrature-type: GLL4

       solver:
         time-marching:
           time-scheme:
             type: Newmark
             dt: 5.0e-4
             nstep: 2000

       simulation-mode:
         forward:
           writer:
             seismogram:
               format: ascii
               directory: OUTPUT_FILES/seismograms

     receivers:
       stations: STATIONS
       angle: 0.0
       seismogram-type:
         - velocity
       nstep_between_samples: 1

     run-setup:
       number-of-processors: 1
       number-of-runs: 1

     databases:
       mesh-database: OUTPUT_FILES/database.bin
       velocity-model:
         file:          MODEL/velocity.dat
         format:        ascii
         interpolation: bilinear
         out-of-bounds: clamp

     sources: sources.yaml

.. code-block:: yaml
   :caption: sources.yaml

   number-of-sources: 1
   sources:
     - force:
         x: 2500.0
         z: 2500.0
         source_surf: false
         angle: 0.0
         vx: 0.0
         vz: 0.0
         Ricker:
           factor: 1.0e10
           tshift: 0.0
           f0: 5.0

.. code-block:: text
   :caption: STATIONS

   S0001  AA  2500.0  0.0  0.0  0.0
   S0002  AA  3500.0  0.0  0.0  0.0
   S0003  AA  4000.0  0.0  0.0  0.0

5. Run
~~~~~~~

.. code-block:: bash

   mkdir -p OUTPUT_FILES/seismograms
   specfem2d -p specfem_config.yaml

Expected output (relevant lines):

.. code-block:: text

   [SF++][  INFO  ]: Reading/injecting GLL model from external reader.
   [SF++][  INFO  ]: Velocity model injection:
   [SF++][  INFO  ]:   File: MODEL/velocity.dat
   [SF++][  INFO  ]: CartesianGrid2D [regular grid]
   [SF++][  INFO  ]:   nx=51  nz=51
   [SF++][  INFO  ]:   x: [0, 5000]  dx=100
   [SF++][  INFO  ]:   z: [0, 5000]  dz=100
   [SF++][  INFO  ]:   total points: 2601
   [SF++][  INFO  ]: Velocity model injection complete.
   [SF++][  INFO  ]: Simulation starts ...

The seismograms in ``OUTPUT_FILES/seismograms/`` will show the reflection from
the z = 2500 m interface.

How properties are converted
------------------------------

The reader converts (Vp, Vs, ρ) from the model file into the native SPECFEM++
storage format for each medium type:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Medium
     - Stored quantities
     - Conversion formula
   * - Elastic (PSV, SH)
     - κ (bulk modulus), μ (shear modulus), ρ (density)
     - κ = ρ (Vp² − 4/3 Vs²),  μ = ρ Vs²
   * - Acoustic
     - ρ⁻¹ (inverse density), κ (bulk modulus)
     - κ = ρ Vp²,  ρ⁻¹ = 1/ρ  (Vs is ignored for fluids)

Parallelism
------------

The injection kernel uses
``Kokkos::MDRangePolicy<DefaultHostExecutionSpace, Rank<3>>`` over the
``(element, iz, ix)`` index space.  This means:

* With an **OpenMP** build the injection is automatically parallelised across
  all available threads.
* With a **CUDA / HIP** build the host injection uses OpenMP on the CPU.
  After all GLL values have been set,
  ``assembly.properties.copy_to_device()`` is called once to transfer the
  full property array to the GPU.

The injection is a one-time startup cost.  Runtime performance of the
time-stepping loop is not affected.

Troubleshooting
---------------

``CartesianGrid2D: cannot open file: ...``
   The path in ``velocity-model.file`` does not exist or is not readable.
   Use an absolute path or ensure the working directory is correct.

``CartesianGrid2D: failed to read binary header from file: ...``
   The binary file is too short or not in the expected format.  Check that
   you wrote the header as ``[int32 NX][int32 NZ][float64 X0 Z0 DX DZ]``.

``CartesianGrid2D: query point (...) is outside the model domain``
   A GLL point lies outside the grid.  Either enlarge the model grid or
   change ``out-of-bounds`` from ``error`` to ``clamp``.

``CartesianGrid2D: expected N data lines, got M``
   The ASCII file has fewer data rows than ``NX × NZ``.  Verify that NX and
   NZ in the header match the actual number of data rows.

``velocity-model: 'file' is a required field``
   The ``velocity-model`` YAML block is present but has no ``file:`` key.

``velocity-model and databases.reader/writer.properties cannot both be specified``
   Remove the legacy ``reader.properties`` / ``writer.properties`` block from
   the ``databases:`` section; use ``velocity-model`` alone.

``WARNING: velocity_model_reader: medium '...' has N element(s) but injection
for this medium type is not yet supported``
   Your mesh contains poroelastic or anisotropic-elastic elements.  Those
   elements retain the mesh-database materials; injection applies only to
   isotropic elastic and acoustic elements.

Frequently asked questions
---------------------------

**Does the model file need to cover the entire mesh domain?**

   Yes (unless you set ``out-of-bounds: clamp``).  With ``clamp``, GLL
   points outside the grid silently receive the nearest boundary value, which
   is usually harmless for small margins.  Use ``out-of-bounds: error`` during
   development to catch accidental coverage gaps.

**Can I mix the velocity-model injection with the mesh-database materials?**

   Not currently.  When ``velocity-model`` is present, every isotropic elastic
   and acoustic element has its properties overridden.  Poroelastic and
   anisotropic elements are untouched (see the warning above).

**What coordinate system should the model file use?**

   The same Cartesian (x, z) coordinate system as the mesh, in metres.  The
   origin and axes are set by the mesher's ``Par_file``.  You can check the
   mesh extent by inspecting the ``OUTPUT_FILES/database.bin`` log or by
   looking at the GLL coordinate ranges printed by SPECFEM++ at startup.

**Is bilinear or nearest-neighbour interpolation more accurate?**

   Bilinear interpolation is more accurate for smooth models (it is
   second-order in the grid spacing).  Nearest-neighbour preserves sharp
   discontinuities (e.g. a flat layer boundary) exactly, without smearing, at
   the cost of a staircase artifact on tilted interfaces.

**My model has a different grid spacing in different regions. Is that supported?**

   Regular grids must have uniform spacing (constant DX, DZ).  For non-uniform
   grids, use the scattered-point format: list every sample point with its
   (x, z, Vp, Vs, Rho) values and the reader will perform nearest-neighbour
   lookup.

**What is the performance impact on a large mesh?**

   Injection is a one-time startup cost proportional to the number of GLL
   points × interpolation cost per point.  For a typical 2D mesh with
   100 000 spectral elements and 25 GLL points each (2.5 M points) and a
   regular-grid model, injection takes a few seconds on a modern workstation
   with OpenMP enabled.  The simulation time-stepping loop is not affected.

See also
--------

* :ref:`homogeneous_example` — basic 2D simulation without velocity-model injection
* :ref:`marmousi_example` — large-scale 2D simulation
* API: ``specfem::io::velocity_model_reader``
* API: ``specfem::io::velocity_model::CartesianGrid2D``
