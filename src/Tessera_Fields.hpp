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

#include "Tessera_RefinementMode.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_MemberTypes.hpp>

#include <cstddef>

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

//! Closure bookkeeping members appended to the face member list in
//! RefinementMode::Conforming ONLY (see Tessera_RefinementMode.hpp):
//!   ClosureParent      GlobalId     gid of the red parent face this closure
//!                                   child retriangulates; invalid_gid on a red
//!                                   face (one that is not a closure child).
//!   ClosureParentVerts GlobalId[3]  the parent's three corner vertex gids, in
//!                                   the parent's winding order.
//! Storing the parent outright on every child means any SINGLE child determines
//! its parent, so the un-close pass needs no sibling lookup.
using ClosureFaceMembers = Cabana::MemberTypes<GlobalId, GlobalId[3]>;

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
// NOTE: `FaceField::UserBegin` is part of the public API — `userFaceField<M>()`
// is `UserBegin + M`. The Conforming-mode closure members are therefore appended
// AFTER the user pack, never inserted among the core members, so `UserBegin`
// stays 5 and every existing user-field index is unchanged in both modes. Their
// indices are computed from the user pack size by
// closureParentField<UserFaceFields>() / closureParentVertsField<...>() below.

// ============================================================================
// Full (core ++ user) member type for each entity kind
// ============================================================================
template <class Scalar, int Dim, class UserVertexFields>
using VertexMemberTypes =
    MemberTypesCat_t<CoreVertexMembers<Scalar, Dim>, UserVertexFields>;

template <class UserEdgeFields>
using EdgeMemberTypes = MemberTypesCat_t<CoreEdgeMembers, UserEdgeFields>;

//! Faces: core ++ user, plus the closure bookkeeping pack in Conforming mode.
//! The layout is  [core | user | (closure)]  so that FaceField::UserBegin and
//! every userFaceField<M>() index are identical in both refinement modes.
template <class UserFaceFields, RefinementMode Mode>
struct FaceMemberTypesImpl
{
    using type = MemberTypesCat_t<CoreFaceMembers, UserFaceFields>;
};

template <class UserFaceFields>
struct FaceMemberTypesImpl<UserFaceFields, RefinementMode::Conforming>
{
    using type =
        MemberTypesCat_t<MemberTypesCat_t<CoreFaceMembers, UserFaceFields>,
                         ClosureFaceMembers>;
};

template <class UserFaceFields,
          RefinementMode Mode = RefinementMode::HangingNode2to1>
using FaceMemberTypes =
    typename FaceMemberTypesImpl<UserFaceFields, Mode>::type;

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

//! Number of USER face fields, i.e. the size of the user pack. Distinct from
//! `face_member_types::size - FaceField::UserBegin`, which in Conforming mode
//! also counts the two trailing closure members: anything iterating the user
//! fields of a face (field copy, I/O) must use THIS count, not the tuple size.
template <class UserFaceFields>
constexpr std::size_t numFaceUserFields()
{
    return UserFaceFields::size;
}

//! Slice index of the ClosureParent member. Valid only when the owning mesh's
//! refinement_mode is RefinementMode::Conforming (in HangingNode2to1 mode the
//! member does not exist and the index is one past the end of the tuple).
template <class UserFaceFields>
constexpr std::size_t closureParentField()
{
    return FaceField::UserBegin + numFaceUserFields<UserFaceFields>();
}

//! Slice index of the ClosureParentVerts member (Conforming mode only).
template <class UserFaceFields>
constexpr std::size_t closureParentVertsField()
{
    return FaceField::UserBegin + numFaceUserFields<UserFaceFields>() + 1;
}

} // namespace Tessera

#endif // TESSERA_FIELDS_HPP
