# StraitFlux Project Analysis

## 1. Project Overview
**StraitFlux** is a Python tool designed to calculate precise oceanic transports (volume, heat, salinity, sea ice) and generate vertical cross-sections across specified straits or lines. It is built to work with CMIP6 model data and supports various curvilinear and unstructured grids (Arakawa A, B, C, E) by standardizing them to an Arakawa C-grid for flux calculations.

## 2. File Structure & Module Breakdown

The core logic resides in the `StraitFlux/` directory.

### File Tree
```
StraitFlux/
├── StraitFlux/
│   ├── __init__.py           # Package initializer
│   ├── functions.py          # Core math: Grid checks, Arakawa transformations, Vertical geometry (dz)
│   ├── functions_VP.py       # Vertical Profile logic: Vector projection, normal calculation
│   ├── indices.py            # Geometric definitions: Strait (line) coordinates -> Grid indices
│   ├── masterscript_line.py  # [MAIN] Transport calculations (Volume, Heat, etc.)
│   ├── masterscript_cross.py # [MAIN] Cross-section generation (Velocities, T, S profiles)
│   └── preprocessing.py      # Data loading, IO validation (xarray/xmip)
├── examples/                 # (Assumed) Download.ipynb, Examples.ipynb
├── README.md                 # Project documentation
└── setup.py                  # Installation script
```

### Module Roles

| File | Role | Key Functions |
|------|------|---------------|
| **`indices.py`** | **Geometry & Grid Mapping**. Defines geographic lines (straits) and maps them to discrete model grid indices (x, y). Handles continuity of the line on the grid. | `def_indices` (presets), `check_availability_indices` (main driver), `prepare_indices` (split into U/V points). |
| **`functions.py`** | **Physics & Math**. Handles grid staggering (Arakawa types) and vertical geometry. Transforms all data to a common C-grid for consistent flux calculation. | `check_Arakawa`, `transform_Arakawa`, `calc_dz_faces`. |
| **`masterscript_line.py`** | **Transport Orchestration**. The entry point for calculating integrated fluxes (Volume [Sv], Heat [PW], etc.). Integrates velocity * area * property. | `transports`. |
| **`masterscript_cross.py`** | **Visualization Orchestration**. The entry point for plotting/calculating 2D vertical cross-sections. Projects velocities onto the section normal. | `vel_projection`, `uvT_projection`, `calc_MOC`. |
| **`functions_VP.py`** | **Projection Logic**. Helper math for cross-sections. Calculates normal vectors and interpolation weights for projecting 3D fields onto 2D sections. | `calc_normvec`, `proj_vec`, `get_nearest_r`. |
| **`preprocessing.py`** | **Data IO**. Wrappers around `xarray` and `xmip` to load CMIP6 data cleanly. | `_preprocess1`, `calc_dxdy`. |

---

## 3. Core Function Analysis

### 3.1 `transports` (in `masterscript_line.py`)
**Functionality**: Calculates the total transport of a property (Volume, Heat, Salt, Ice) across a defined strait over time.

**Workflow**:
1.  **Define Geometry**: Calls `check_availability_indices` to get the grid indices $(x, y)$ that approximate the geographic line.
2.  **Load Data**: Loads $u, v, T, S$ and cell thickness ($dz$).
3.  **Grid Standardization**: Calls `functions.check_Arakawa` and `transform_Arakawa` to ensure velocities and thicknesses are defined at the cell faces (C-grid).
    -   *Why?* Accurate flux calculation requires velocity perpendicular to the face area.
4.  **Property Calculation**:
    -   **Volume**: $Flux = u \cdot dy \cdot dz$ (or $v \cdot dx \cdot dz$)
    -   **Heat**: $Flux = Volume\_Flux \cdot \rho \cdot C_p \cdot (T - T_{ref})$
    -   **Salt**: $Flux = Volume\_Flux \cdot \rho \cdot S$
5.  **Integration**:
    -   Sums over depth (`lev`).
    -   Selects only the grid faces marked by the indices.
    -   Applies directional signs (+/-) depending on whether the flow is North/East or South/West relative to the line direction.
    -   Sums along the line.

### 3.2 `def_indices` (in `indices.py`)
**Functionality**: Returns the latitude and longitude arrays defining a strait.
**Input**: Strait name (e.g., 'Fram', 'Bering') or custom coordinates.
**Output**: Arrays of `lat` and `lon` points.
**Logic**: Contains hardcoded coordinates for standard oceanographic sections. If a custom path is needed, it interpolates points between start/end coordinates.

### 3.3 `calc_dz_faces` (in `functions.py`)
**Functionality**: Computes the vertical thickness of grid cells at the **velocity faces** (u and v points), not just the cell centers.
**Logic**:
-   Cell thickness (`thkcello`) is usually given at the cell center.
-   To calculate flux through a face, we need the thickness at that face.
-   For Arakawa C-grids:
    -   Thickness at U-face (East/West) $\approx$ min/interpolated from adjacent centers $(i, j)$ and $(i, j+1)$.
    -   Thickness at V-face (North/South) $\approx$ min/interpolated from adjacent centers $(i, j)$ and $(i+1, j)$.
-   This ensures that if a cell is land (thickness 0), the flow through the adjacent face is correctly zeroed out.

---

## 4. Data Flow & Architecture

```mermaid
graph TD
    User[User Input\n(Strait Name, Model, Year)] --> Master[transports()]
    
    subgraph "Geometry Phase"
        Master --> Indices[indices.py]
        Indices --> Def[def_indices]
        Def --> Points[Lat/Lon Points]
        Points --> GridMap[check_availability_indices]
        GridMap --> IndexList[Grid Indices (x,y)\nU/V Face Flags]
    end
    
    subgraph "Data Loading Phase"
        Master --> Pre[preprocessing.py]
        Pre --> RawData[Load NetCDF (u, v, T, dz)]
    end
    
    subgraph "Physics Phase"
        Master --> Func[functions.py]
        Func --> Check[check_Arakawa]
        RawData --> Transform[transform_Arakawa]
        Check --> Transform
        Transform --> CGridData[Standardized Data\n(u_face, v_face)]
        Transform --> CalcDz[calc_dz_faces]
        CalcDz --> FaceAreas
    end
    
    subgraph "Calculation Phase"
        CGridData & FaceAreas --> FluxCalc[Compute Flux Element\n(u * Area * Property)]
        FluxCalc -- Filter by Indices --> LineFluxes
        LineFluxes -- Integrate --> TotalTransport
    end
    
    TotalTransport --> Output[NetCDF File\n(Time Series)]
```

### Key Technical Concepts
1.  **Arakawa C-Grid**: The standard grid structure used for internal calculations. Velocities are perpendicular to cell faces, making flux calculation ($u \times Area$) straightforward.
2.  **Curvilinear Coordinates**: The code relies on the model providing `dyu` (width of u-face) and `dxv` (width of v-face) or calculates them. It does **not** assume a regular lat-lon grid.
3.  **Zig-Zag Line Approximation**: The geographic line is approximated by a "staircase" of grid cell faces. The `indices.py` logic ensures this path is continuous and water-tight (no leaks).

---

## 5. Usage Example

To calculate **Heat Transport** across the **Fram Strait**:

```python
from StraitFlux.masterscript_line import transports

transports(
    product='heat',          # Calculate Heat Transport
    strait='Fram',           # Use predefined Fram Strait coordinates
    model='MPI-ESM1-2-HR',   # Model Name
    time_start=2000,         # Start Year
    time_end=2001,           # End Year
    file_u='/path/to/uo*.nc',
    file_v='/path/to/vo*.nc',
    file_t='/path/to/thetao*.nc',
    file_z='/path/to/thkcello*.nc',
    path_save='./output/',
    path_indices='./indices/',
    path_mesh='./mesh/'
)
```

**Background on External Concepts**:
-   **CMIP6**: The standard format for data output in the current generation of climate models.
-   **Arakawa Grids**: A classification of how variables are distributed on a grid. "C-grid" is most common for ocean models (like NEMO, MITgcm) because it naturally represents conservation of mass/volume.
