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

#include <cstddef>
#include <cstdint>
#include <memory>

#include "exceptions.hpp"
#include "lp_norm.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/norm.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector lp_norm(const Node& node)
                {
                    const std::shared_ptr<ngraph::Node> data{node.get_ng_inputs().at(0)};
                    const std::int64_t p_norm{node.get_attribute_value<std::int64_t>("p", 2)};

                    const std::int64_t axis{node.get_attribute_value<std::int64_t>("axis", -1)};
                    const std::size_t valid_axis =
                        common::validate_axis(node, axis, data->get_shape().size());

                    ASSERT_VALID_ARGUMENT(node, p_norm == 1 || p_norm == 2)
                        << "Invalid `p` attribute value: " << p_norm
                        << "Only normalization of 1st or 2nd order is supported.";

                    std::shared_ptr<ngraph::Node> norm = ngraph::builder::lp_norm(
                        data, AxisSet{valid_axis}, static_cast<std::size_t>(p_norm));

                    const auto target_shape = ngraph::op::Constant::create(
                        element::i64, Shape{data->get_shape().size()}, data->get_shape());

                    std::vector<size_t> axes_values(data->get_shape().size());
                    std::iota(axes_values.begin(), axes_values.end(), 0);
                    axes_values.erase(axes_values.begin() + valid_axis);

                    const auto axes_mapping = ngraph::op::Constant::create(
                        element::i64, Shape{axes_values.size()}, axes_values);

                    norm = std::make_shared<default_opset::Broadcast>(
                        norm, target_shape, axes_mapping);

                    return {std::make_shared<default_opset::Divide>(data, norm)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
