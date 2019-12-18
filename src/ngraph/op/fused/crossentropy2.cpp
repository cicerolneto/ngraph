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
#include "ngraph/op/fused/crossentropy2.hpp"

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/util/broadcasting.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::CrossEntropy2::type_info;

op::CrossEntropy2::CrossEntropy2(const Output<Node>& arg1,
                                 const Output<Node>& arg2,
                                 bool soft_label,
                                 int64_t ignore_index)
    : FusedOp({arg1, arg2})
    , m_soft_label(soft_label)
    , m_ignore_index(ignore_index)
{
    constructor_validate_and_infer_types();
}

static AxisVector get_axis_vector(size_t rank)
{
    AxisVector axis_vector;

    for (size_t i = 0; i < rank; i++)
    {
        axis_vector.push_back(i);
    }
    return axis_vector;
}

static Shape get_result_shape(Shape& target_shape, int start, int end)
{
    Shape result;
    for (size_t i = start; i < end; i++)
    {
        result.push_back(target_shape[i]);
    }
    return result;
}

static Output<Node> get_2d_tensor(Output<Node> node)
{
    if (node.get_shape().size() == 2)
    {
        return node;
    }
    Shape node_shape = node.get_shape();
    size_t rank = node_shape.size();
    Shape result_shape{(shape_size(node_shape) / node_shape[rank - 1]), node_shape[rank - 1]};

    auto reshape = std::make_shared<ngraph::op::Reshape>(node, get_axis_vector(rank), result_shape);
    return reshape;
}

static std::shared_ptr<Node> expand_shape(std::shared_ptr<Node> result, Output<Node> original)
{
    Shape result_shape = result->get_shape();
    Shape original_shape = original.get_shape();

    if (result_shape == original_shape && result_shape.size() == 2)
    {
        return result;
    }
    size_t original_rank = original_shape.size();
    size_t result_rank = result_shape.size();

    // expand the first dimension of the computed result to match the original tensor shape
    Shape new_shape = get_result_shape(original_shape, 0, original_rank - 1);

    // restore the last dimension of computed result
    new_shape.push_back(result_shape[result_rank - 1]);

    if (new_shape.size() != original_shape.size())
    {
        throw ngraph_error(
            "CrossEntropy shape size mismatch in restoring the original tensor shape");
    }
    auto reshape = std::make_shared<ngraph::op::Reshape>(result, AxisVector{0, 1}, new_shape);
    return reshape;
}

// create mask based on ignore_index
static std::shared_ptr<ngraph::Node>
    create_mask(Output<Node> labels, Output<Node> input, int64_t ignore_index)
{
    auto mask_constant =
        ngraph::op::Constant::create(labels.get_element_type(), labels.get_shape(), {ignore_index});
    auto not_equal = std::make_shared<ngraph::op::NotEqual>(labels, mask_constant);
    auto convert = std::make_shared<ngraph::op::Convert>(not_equal, input.get_element_type());
    return convert;
}

NodeVector op::CrossEntropy2::decompose_op() const
{
    auto input = get_2d_tensor(input_value(0));
    auto labels = get_2d_tensor(input_value(1));
    auto reduction_axis = input.get_shape().size() - 1;
    /*
        if (get_input_partial_shape(1).is_dynamic())
        {
            NODE_VALIDATION_CHECK(
                this,
                PartialShape::merge_into(inputs_shape_scheme, this_input_shape),
                "Argument shapes are inconsistent; they must have the same rank, and must have ",
                "equal dimension everywhere except on the concatenation axis (axis ",
                axis,
                ").");
        }

        if (get_inpit_partial_shape(2).is_dynamic())
        {
            NODE_VALIDATION_CHECK(
                this,
                PartialShape::merge_into(inputs_shape_scheme, this_input_shape),
                "Argument shapes are inconsistent; they must have the same rank, and must have ",
                "equal dimension everywhere except on the concatenation axis (axis ",
                axis,
                ").");
        }
    */
    auto reshape = [&](const Output<Node>& input, ngraph::Shape shape) {
        std::vector<size_t> input_order(input.get_shape().size());
        std::iota(std::begin(input_order), std::end(input_order), 0);
        std::shared_ptr<ngraph::Node> reshape =
            std::make_shared<op::Reshape>(input, ngraph::AxisVector(input_order), shape);
        return reshape;
    };

    auto create_xe = [&](const Output<Node>& one_hot, const Output<Node>& input) {
        auto node_log = std::make_shared<op::Log>(input);
        auto node_mul = one_hot * node_log;
        auto node_sum =
            std::make_shared<op::Sum>(node_mul, AxisSet{static_cast<size_t>(reduction_axis)});
        auto input_shape = input.get_shape();
        input_shape.back() = 1;
        auto node_sum_reshape = reshape(node_sum, input_shape);
        return -node_sum_reshape;
    };

    auto one_hot_shape = input.get_shape();
    auto rank = one_hot_shape.size() - 1;
    auto label_shape = labels.get_shape();
    std::shared_ptr<ngraph::Node> one_hot_labels;

    if (label_shape.back() == 1 && label_shape.size() > 1)
    {
        label_shape.pop_back();
        auto label_reshape = reshape(labels, label_shape);
        one_hot_labels = std::make_shared<op::OneHot>(label_reshape, one_hot_shape, rank);
    }
    else
    {
        one_hot_labels = std::make_shared<op::OneHot>(labels, one_hot_shape, rank);
    }

    auto input_type = input.get_element_type();
    one_hot_labels = std::make_shared<op::Convert>(one_hot_labels, input_type);
    auto xe = create_xe(one_hot_labels, input);
    auto mask = create_mask(labels, input, m_ignore_index);
    mask = std::make_shared<op::Convert>(mask, input_type);
    xe = xe * mask;
    auto node_sum = std::make_shared<op::Sum>(one_hot_labels * input, ngraph::AxisSet{rank});
    auto sum_reshape = reshape(node_sum, mask->get_shape());
    auto matchx = mask * sum_reshape;

    return {matchx};
}

shared_ptr<Node> op::CrossEntropy2::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<CrossEntropy2>(new_args.at(0), new_args.at(1), m_soft_label, m_ignore_index);
}

void op::CrossEntropy2::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");
    set_output_type(0, get_input_element_type(0), PartialShape::dynamic());

    if (is_dynamic())
    {
        return;
    }
}

constexpr NodeTypeInfo op::CrossEntropy2Backprop::type_info;

op::CrossEntropy2Backprop::CrossEntropy2Backprop(const Output<Node>& input,
                                                 const Output<Node>& labels,
                                                 const Output<Node>& delta,
                                                 bool soft_label,
                                                 int64_t ignore_index)
    : FusedOp({input, labels, delta})
    , m_soft_label(soft_label)
    , m_ignore_index(ignore_index)
{
    constructor_validate_and_infer_types();
}

void op::CrossEntropy2Backprop::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");
    set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
}

shared_ptr<Node> op::CrossEntropy2Backprop::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<CrossEntropy2Backprop>(
        new_args.at(0), new_args.at(1), new_args.at(2), m_soft_label, m_ignore_index);
}

NodeVector op::CrossEntropy2Backprop::decompose_op() const
{
    auto matchx = get_2d_tensor(input_value(0));
    auto label = get_2d_tensor(input_value(1));
    auto x = get_2d_tensor(input_value(2));
    auto dy = get_2d_tensor(input_value(3));

    auto reshape = [&](const Output<Node>& input, ngraph::Shape shape) {
        std::vector<size_t> input_order(input.get_shape().size());
        std::iota(std::begin(input_order), std::end(input_order), 0);
        std::shared_ptr<ngraph::Node> reshape =
            std::make_shared<op::Reshape>(input, ngraph::AxisVector(input_order), shape);
        return reshape;
    };

    auto matchx_shape = matchx.get_shape();
    auto label_shape = label.get_shape();
    auto x_shape = x.get_shape();
    auto dy_shape = dy.get_shape();
    if (matchx_shape.back() == 1 && matchx_shape.size() > 1)
    {
        matchx_shape.pop_back();
        matchx = reshape(matchx, matchx_shape);
    }

    if (label_shape.back() == 1 && label_shape.size() > 1)
    {
        label_shape.pop_back();
        label = reshape(label, label_shape);
    }

    if (x_shape.back() == 1 && x_shape.size() > 1)
    {
        x_shape.pop_back();
        x = reshape(x, x_shape);
    }

    if (dy_shape.back() == 1 && dy_shape.size() > 1)
    {
        dy_shape.pop_back();
        dy = reshape(dy, dy_shape);
    }

    auto rank = x_shape.size();
    auto x_type = x.get_element_type();
    auto one_hot = std::make_shared<op::OneHot>(label, x_shape, rank);

    std::shared_ptr<ngraph::Node> one_hot_labels = std::make_shared<op::Convert>(one_hot, x_type);

    auto mask = create_mask(label, x, m_ignore_index);
    mask = std::make_shared<op::Convert>(mask, x_type);

    auto zero = op::Constant::create(matchx.get_element_type(), matchx.get_shape(), {0});
    auto one = op::Constant::create(matchx.get_element_type(), matchx.get_shape(), {1});

    auto is_zero = std::make_shared<op::Equal>(matchx, zero);
    matchx = std::make_shared<ngraph::op::Select>(is_zero, one, matchx);

    auto dy_bcast =
        std::make_shared<ngraph::op::Broadcast>(mask * dy, x_shape, ngraph::AxisSet{rank - 1});

    auto matchx_bcast =
        std::make_shared<ngraph::op::Broadcast>(matchx, x_shape, ngraph::AxisSet{rank - 1});

    auto xe_grad = -dy_bcast * one_hot_labels / matchx_bcast;
    return {expand_shape(xe_grad, input_value(0))};
}
