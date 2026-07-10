#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/element_types.hpp"
#include "aero/cfd/mesh_io.hpp"
#include "aero/cfd/mesh_io_cgns.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#ifdef WITH_CGNS
#include <cgnslib.h>
#endif

namespace aerosp {
namespace aero {
namespace cfd {

namespace {

#ifdef WITH_CGNS

ElementType cgns_elem_to_type(int elem_type) {
    switch (elem_type) {
        case TETRA_4:   return ElementType::TET4;
        case HEXA_8:    return ElementType::HEX8;
        case PENTA_6:   return ElementType::PENTA6;
        case PYRA_5:    return ElementType::PYRAMID5;
        default:        return ElementType::TET4;
    }
}

int expected_elem_nodes(int elem_type) {
    switch (elem_type) {
        case TETRA_4:   return 4;
        case HEXA_8:    return 8;
        case PENTA_6:   return 6;
        case PYRA_5:    return 5;
        case TRI_3:     return 3;
        case QUAD_4:    return 4;
        default:        return 0;
    }
}

BoundaryKind cgns_bc_to_kind(int bc_type) {
    switch (bc_type) {
        case BCWallInviscid: return BoundaryKind::SlipWall;
        case BCWallViscous:  return BoundaryKind::NoSlipWall;
        case BCFarfield:     return BoundaryKind::Farfield;
        case BCSymmetryPlane: return BoundaryKind::Symmetry;
        case BCInflow:
        case BCOutflow:
        case BCInflowSubsonic:
        case BCOutflowSubsonic:
        case BCInflowSupersonic:
        case BCOutflowSupersonic:
            return BoundaryKind::Farfield;
        default:
            return BoundaryKind::Farfield;
    }
}

#endif // WITH_CGNS

} // namespace

bool read_mesh_cgns(const std::string& path, CfdMesh& mesh, std::string* err) {
#ifdef WITH_CGNS
    mesh = CfdMesh{};

    int fn;
    if (cg_open(path.c_str(), CG_MODE_READ, &fn) != CG_OK) {
        if (err) {
            char buf[256];
            cg_get_error(buf);
            *err = std::string("CGNS open failed: ") + buf;
        }
        return false;
    }

    int nbases;
    cg_nbases(fn, &nbases);
    if (nbases < 1) {
        if (err) *err = "CGNS file has no base";
        cg_close(fn);
        return false;
    }

    char basename[33];
    int celldim, physdim;
    cg_base_read(fn, 1, basename, &celldim, &physdim);
    if (celldim != 3 || physdim != 3) {
        if (err) *err = "CGNS: only 3D meshes supported";
        cg_close(fn);
        return false;
    }

    int nzones;
    cg_nzones(fn, 1, &nzones);
    if (nzones < 1) {
        if (err) *err = "CGNS file has no zones";
        cg_close(fn);
        return false;
    }

    for (int Z = 1; Z <= nzones; ++Z) {
        char zonename[33];
        cgsize_t size[9];
        cg_zone_read(fn, 1, Z, zonename, size);

        ZoneType_t zonetype;
        cg_zone_type(fn, 1, Z, &zonetype);
        if (zonetype != Unstructured) {
            if (err) *err = std::string("CGNS zone '") + zonename + "' is not unstructured";
            cg_close(fn);
            return false;
        }

        int nnodes = static_cast<int>(size[0]);

        // Read node coordinates
        int ncoords;
        cg_ncoords(fn, 1, Z, &ncoords);
        if (ncoords < 3) {
            if (err) *err = "CGNS: need at least 3 coordinate arrays";
            cg_close(fn);
            return false;
        }

        std::vector<Real> xs(nnodes), ys(nnodes), zs(nnodes);
        DataType_t datatype;

        cg_coord_info(fn, 1, Z, 1, &datatype);
        if (datatype == RealDouble) {
            std::vector<double> tmp(nnodes);
            cg_coord_read(fn, 1, Z, "CoordinateX", RealDouble, &tmp[0]);
            for (int i = 0; i < nnodes; ++i) xs[i] = static_cast<Real>(tmp[i]);
            cg_coord_read(fn, 1, Z, "CoordinateY", RealDouble, &tmp[0]);
            for (int i = 0; i < nnodes; ++i) ys[i] = static_cast<Real>(tmp[i]);
            cg_coord_read(fn, 1, Z, "CoordinateZ", RealDouble, &tmp[0]);
            for (int i = 0; i < nnodes; ++i) zs[i] = static_cast<Real>(tmp[i]);
        } else {
            cg_coord_read(fn, 1, Z, "CoordinateX", RealSingle, &xs[0]);
            cg_coord_read(fn, 1, Z, "CoordinateY", RealSingle, &ys[0]);
            cg_coord_read(fn, 1, Z, "CoordinateZ", RealSingle, &zs[0]);
        }

        int base_offset = static_cast<int>(mesh.nodes.size());
        mesh.nodes.reserve(mesh.nodes.size() + nnodes);
        for (int i = 0; i < nnodes; ++i)
            mesh.nodes.push_back({xs[i], ys[i], zs[i]});

        // Read element sections
        int nsections;
        cg_nsections(fn, 1, Z, &nsections);

        for (int S = 1; S <= nsections; ++S) {
            char secname[33];
            ElementType_t elem_type;
            cgsize_t start, end;
            int nbndry, parent_flag;
            cg_section_read(fn, 1, Z, S, secname, &elem_type, &start, &end, &nbndry, &parent_flag);

            int nelem = static_cast<int>(end - start + 1);
            int nnodes_per_elem = expected_elem_nodes(elem_type);
            if (nnodes_per_elem == 0) continue;

            int total_conn = nnodes_per_elem * nelem;
            std::vector<cgsize_t> conn(total_conn);
            cgsize_t data_size;
            cg_elements_read(fn, 1, Z, S, &conn[0], &data_size);

            // Determine if this is a volume or boundary element
            bool is_volume = (elem_type == TETRA_4 || elem_type == HEXA_8 ||
                              elem_type == PENTA_6 || elem_type == PYRA_5);
            bool is_face = (elem_type == TRI_3 || elem_type == QUAD_4);

            if (is_volume) {
                ElementType etype = cgns_elem_to_type(elem_type);
                for (int e = 0; e < nelem; ++e) {
                    CfdCell cell;
                    cell.type = etype;
                    for (int j = 0; j < nnodes_per_elem && j < 8; ++j) {
                        int node_id = static_cast<int>(conn[e * nnodes_per_elem + j]) - 1 + base_offset;
                        cell.node[j] = node_id;
                    }
                    mesh.cells.push_back(cell);
                }
            }
        }

        // Read boundary conditions
        int nbocos;
        cg_nbocos(fn, 1, Z, &nbocos);

        for (int BC = 1; BC <= nbocos; ++BC) {
            char boconame[33];
            BCType_t bocotype;
            cgsize_t ptset_type, npnts, normal_list_size;
            cgsize_t normal_list_flag, normal_dir_flag;
            cg_boco_read(fn, 1, Z, BC, boconame, &bocotype, &ptset_type,
                        &npnts, nullptr, &normal_list_size,
                        &normal_list_flag, &normal_dir_flag);

            BoundaryKind bkind = cgns_bc_to_kind(bocotype);

            // Read boundary face elements
            if (ptset_type == Element_t || ptset_type == FaceCenter) {
                std::vector<cgsize_t> face_conn(npnts);
                cg_boco_read(fn, 1, Z, BC, boconame, &bocotype, &ptset_type,
                            &npnts, &face_conn[0], &normal_list_size,
                            &normal_list_flag, &normal_dir_flag);

                // face_conn contains local element/section references.
                // We need to look up the actual element from the sections.
                // For Element_t ptSet: face_conn contains (section_idx, elem_idx) pairs.
                // We'll add the boundary faces later via rebuild_faces + marker override.
            }
        }
    }

    // Build faces from cells
    rebuild_mesh_faces(mesh);

    // Apply boundary markers from CGNS boundary conditions
    // (simplified: rebuild_faces classifies walls from geometry,
    //  CGNS markers could override here if needed)

    cg_close(fn);
    return true;

#else // WITH_CGNS not defined
    (void)mesh;
    if (err) *err = "CGNS support not compiled in (set AEROSIM_USE_CGNS=ON)";
    return false;
#endif
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
