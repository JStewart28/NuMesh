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

#ifndef TESSERA_FIELDS_HPP
#define TESSERA_FIELDS_HPP

#include "Tessera_Types.hpp"

#include <Cabana_MemberTypes.hpp>

namespace Tessera
{

// ============================================================================
// Field pack model
// ============================================================================
//
// Each entity kind stores its data in a single Cabana AoSoA whose member list
// is the concatenation of:
//   (1) a fixed set of CORE members (identity, ownership, refinement level, and
//       connectivity) that Tessera itself manages, followed by
//   (2) an arbitrary USER field pack declared at compile time by the caller.
//
// The user pack is an ordinary Cabana::MemberTypes list, so users get typed,
// vectorizable slices in their kernels and every user field is haloed/migrated
// with its entity automatically (the comm layer is generic over the whole AoSoA
// tuple — no per-field plumbing). Position is a CORE vertex field of type
// Scalar[Dim] because the partitioner centroid, I/O, and local operators all
// require it.
//
// User packs are spelled with these thin aliases for readability at the call
// site, e.g.  VertexFields<Scalar[2] /*vorticity*/>,  FaceFields<Scalar>:
template <class... Ts>
using VertexFields = Cabana::MemberTypes<Ts...>;
template <class... Ts>
using EdgeFields = Cabana::MemberTypes<Ts...>;
template <class... Ts>
using FaceFields = Cabana::MemberTypes<Ts...>;

// ----------------------------------------------------------------------------
// Concatenate two Cabana::MemberTypes lists: core ++ user.
// ----------------------------------------------------------------------------
template <class A, class B>
struct MemberTypesCat;

template <class... A, class... B>
struct MemberTypesCat<Cabana::MemberTypes<A...>, Cabana::MemberTypes<B...>>
{
    using type = Cabana::MemberTypes<A..., B...>;
};

template <class A, class B>
using MemberTypesCat_t = typename MemberTypesCat<A, B>::type;

// ============================================================================
// Core member layouts (integer/topology data Tessera manages)
// ============================================================================
//
// Slice indices for the core members are named in the *Field namespaces below.
// User fields begin at index `UserBegin` for that entity kind.

//! Vertex core: gid, owner rank, flags, position[Dim].
template <class Scalar, int Dim>
using CoreVertexMembers =
    Cabana::MemberTypes<GlobalId, Rank, std::int32_t, Scalar[Dim]>;

//! Edge core: gid, owner rank, level, endpoint vertex gids v[2],
//! incident face gids f[2] (invalid_gid where absent, e.g. a boundary edge).
using CoreEdgeMembers =
    Cabana::MemberTypes<GlobalId, Rank, Level, GlobalId[2], GlobalId[2]>;

//! Face core: gid, owner rank, level, corner vertex gids v[3], edge gids e[3].
using CoreFaceMembers =
    Cabana::MemberTypes<GlobalId, Rank, Level, GlobalId[3], GlobalId[3]>;

// ----------------------------------------------------------------------------
// Named core slice indices. User fields follow at `UserBegin`.
// ----------------------------------------------------------------------------
namespace VertexField
{
enum : std::size_t
{
    Gid = 0,
    Owner = 1,
    Flags = 2,
    Position = 3,
    UserBegin = 4
};
}

namespace EdgeField
{
enum : std::size_t
{
    Gid = 0,
    Owner = 1,
    Level = 2,
    Verts = 3, // GlobalId[2]
    Faces = 4, // GlobalId[2]
    UserBegin = 5
};
}

namespace FaceField
{
enum : std::size_t
{
    Gid = 0,
    Owner = 1,
    Level = 2,
    Verts = 3, // GlobalId[3]
    Edges = 4, // GlobalId[3]
    UserBegin = 5
};
}

// ============================================================================
// Full (core ++ user) member type for each entity kind
// ============================================================================
template <class Scalar, int Dim, class UserVertexFields>
using VertexMemberTypes =
    MemberTypesCat_t<CoreVertexMembers<Scalar, Dim>, UserVertexFields>;

template <class UserEdgeFields>
using EdgeMemberTypes = MemberTypesCat_t<CoreEdgeMembers, UserEdgeFields>;

template <class UserFaceFields>
using FaceMemberTypes = MemberTypesCat_t<CoreFaceMembers, UserFaceFields>;

//! Index of the M-th user field within the full member list of an entity kind.
//! Usage: mesh.vertexSlice<Tessera::userVertexField<0>()>()
template <std::size_t M>
constexpr std::size_t userVertexField()
{
    return VertexField::UserBegin + M;
}
template <std::size_t M>
constexpr std::size_t userEdgeField()
{
    return EdgeField::UserBegin + M;
}
template <std::size_t M>
constexpr std::size_t userFaceField()
{
    return FaceField::UserBegin + M;
}

} // namespace Tessera

#endif // TESSERA_FIELDS_HPP
