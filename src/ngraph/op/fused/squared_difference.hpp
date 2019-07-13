//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Calculates an element-wise squared difference between two tensors
        ///
        /// y[i] = (x1[i] - x2[i])^2
        class SquaredDifference : public ngraph::op::util::FusedOp
        {
        public:
            /// \brief Constructs the squared difference operation.
            ///
            /// \param x1 First input tensor
            /// \param x2 Second input tensor
            SquaredDifference(const std::shared_ptr<ngraph::Node>& x1,
                              const std::shared_ptr<ngraph::Node>& x2);

            virtual NodeVector decompose_op() const override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
        };
    }
}
