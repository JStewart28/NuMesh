/****************************************************************************
 * Copyright (c) 2024, JStewart28                                           *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Tessera library. Tessera is distributed under a *
 * BSD 3-Clause license. For the licensing terms see the LICENSE file in   *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef TESSERA_HPP
#define TESSERA_HPP

// Umbrella header for the Tessera distributed unstructured triangle-mesh
// library.

#include "Tessera_AllToAllV.hpp"
#include "Tessera_CsrAdjacency.hpp"
#include "Tessera_Distribute.hpp"
#include "Tessera_EdgeSplit.hpp"
#include "Tessera_EditFamily.hpp"
#include "Tessera_FaceAdjacency.hpp"
#include "Tessera_FieldReduce.hpp"
#include "Tessera_Fields.hpp"
#include "Tessera_Geometry.hpp"
#include "Tessera_HDF5Reader.hpp"
#include "Tessera_HDF5Writer.hpp"
#include "Tessera_HaloExchange.hpp"
#include "Tessera_HaloRebuild.hpp"
#include "Tessera_HaloScatterAdd.hpp"
#include "Tessera_Icosphere.hpp"
#include "Tessera_IoCommon.hpp"
#include "Tessera_MarkQuality.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_MeshBuilder.hpp"
#include "Tessera_MeshMigrate.hpp"
#include "Tessera_Migrate.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_Reduction.hpp"
#include "Tessera_Refine.hpp"
#include "Tessera_RefineClosure.hpp"
#include "Tessera_RefineParallel.hpp"
#include "Tessera_RefinePolicy.hpp"
#include "Tessera_RefinementMode.hpp"
#include "Tessera_RegisteredBufferPool.hpp"
#include "Tessera_Stencil.hpp"
#include "Tessera_Types.hpp"
#include "Tessera_Xdmf.hpp"
#include "Tessera_Zoltan2Balancer.hpp"

#endif // TESSERA_HPP
