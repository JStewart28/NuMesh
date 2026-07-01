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

#ifndef TESSERA_REFINE_POLICY_HPP
#define TESSERA_REFINE_POLICY_HPP

#include "Tessera_Fields.hpp"

#include <cstddef>

namespace Tessera
{

// ============================================================================
// Refinement interpolation policy
// ============================================================================
//
// When a face is refined (Tessera_Refine.hpp), a new midpoint vertex is inserted
// on each split edge. Its coordinates and every user field must be assigned from
// the two edge endpoints. That assignment is *pluggable*: the refinement engine
// owns the topology (which vertices/faces/edges are created and how they connect);
// the policy owns only the numeric blend of a midpoint from its two parents. This
// separation lets a curvature-aware geometric scheme (e.g. modified Butterfly) or
// a physics-correct field rule (e.g. a vorticity/sheet-strength conservation rule)
// be substituted without touching the refinement machinery.
//
// A policy provides two hooks:
//
//   void interpolatePosition( Scalar* mid, const Scalar* a, const Scalar* b,
//                             int dim ) const;
//       Fill `mid[0..dim)` (the core Scalar[Dim] position) from endpoints
//       `a`, `b`. The default is the linear edge midpoint 0.5*(a+b). Note AMR
//       does NOT project onto the sphere — projection is an icosphere-generation
//       step only, so a refined midpoint is a plain average, not unit length.
//
//   template <std::size_t M> Scalar interpolateVertexField( Scalar a,
//                                                            Scalar b ) const;
//       Blend one scalar component of the vertex user field at ABSOLUTE member
//       index `M` (i.e. M runs over VertexField::UserBegin .. last member). For
//       an array-valued field (Scalar[N]) the engine calls this once per
//       component. The default is the linear average 0.5*(a+b).
//
// Per-field override pattern: derive from DefaultRefinePolicy and shadow
// interpolateVertexField, dispatching on M with `if constexpr` so exactly one
// field's rule changes while every other field falls through to the base:
//
//   struct MyPolicy : Tessera::DefaultRefinePolicy<double>
//   {
//       template <std::size_t M>
//       double interpolateVertexField( double a, double b ) const
//       {
//           if constexpr ( M == Tessera::VertexField::UserBegin + 0 )
//               return conserve_vorticity( a, b );      // override field 0
//           else
//               return Tessera::DefaultRefinePolicy<double>::
//                   template interpolateVertexField<M>( a, b );
//       }
//   };
//
// The engine calls the policy on the host (refinement derivation is host-side, as
// for the mesh builder), so a policy may use ordinary host code.

template <class Scalar>
struct DefaultRefinePolicy
{
    using scalar_type = Scalar;

    //! Linear edge midpoint: mid[d] = 0.5 * ( a[d] + b[d] ) for d in [0,dim).
    void interpolatePosition( Scalar* mid, const Scalar* a, const Scalar* b,
                              int dim ) const
    {
        for ( int d = 0; d < dim; ++d )
            mid[d] = static_cast<Scalar>( 0.5 ) * ( a[d] + b[d] );
    }

    //! Linear average of a single scalar component of vertex user field #M.
    template <std::size_t M>
    Scalar interpolateVertexField( Scalar a, Scalar b ) const
    {
        return static_cast<Scalar>( 0.5 ) * ( a + b );
    }
};

} // namespace Tessera

#endif // TESSERA_REFINE_POLICY_HPP
