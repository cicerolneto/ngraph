
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

// NOTE: This file follows nGraph format style.
// Follows nGraph naming convention for public APIs only, else MLIR naming convention.

#include "fused_ops_pattern.h.inc"

namespace
{
    class NgDialectFusedOpsPass : public mlir::ModulePass<NgDialectFusedOpsPass>
    {
    public:
        NgDialectFusedOpsPass(mlir::ModuleOp module, 
                              mlir::MLIRContext* context)
            : m_module(module)
            , m_context(context)
        {
        }

        NgDialectFusedOpsPass (const NgDialectFusedOpsPass& obj);

        private:
	    void runOnModule() override;
            mlir::ModuleOp m_module;
            mlir::MLIRContext* m_context;

    };
}

NgDialectFusedOpsPass::NgDialectFusedOpsPass(const NgDialectFusedOpsPass& obj)
    : m_module(obj.m_module)
    , m_context(obj.m_context)
{
}

void NgDialectFusedOpsPass::runOnModule()
{
   // Create type converter and initialize conversion patterns.
   NGraphTypeConverter converter;
   OwningRewritePatternList patterns;

   populateWithGenerated(m_context, patterns);

   ConversionTarget target(m_context);

   target.addLegalDialect<AffineOpsDialect, StandardOpsDialect>();
   target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
   target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
       // FuncOp is legal only if types have been converted to Std types.
       return typeConverter.isSignatureLegal(op.getType());
   });

   // Gather functions to be processed. Note that new functions will be added to module as part
   // of the function signature conversion so we have to collect the original ones before hand.
   SmallVector<FuncOp, 2> origFuncOps(m_module.getOps<FuncOp>());

   for (auto origFunc : origFuncOps)
   {
       // TODO: Workaround for findOutputValues and buildOutputDefs. See NGCPU-470.
       funcName = origFunc.getName();

       // Capture output values by looking for the Return and grabbing the values the order of
       // the returned values matches the order of the lowered func signature for results. This
       // is used to find the arg_id that a defined value maps to if it is an output.
       findOutputValues();

       // NOTE: Function signature conversion creates a new FuncOp that is inserted in the
       // module. References the original FuncOp are no longer valid after this point.
       if (failed(applyFullConversion(origFunc, target, std::move(patterns), &converter)))
       {
           emitError(mlir::UnknownLoc::get(m_context), "Error fusing MatMulBias \n");
           signalPassFailure();
       }

       // TODO: Encode no alias attribute as part of the function signature conversion or as a
       // separate rewrite pattern. Retrieve new function after signature conversion.
       insertNoAliasArgAttrs();
   } 
}
