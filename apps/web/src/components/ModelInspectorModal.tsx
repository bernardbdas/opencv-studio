import React from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { Info, X, Cpu, Layers, Download, CheckCircle2, BookOpen } from 'lucide-react';
import { MODEL_METADATA_REGISTRY, DetailedModelSpec } from '@opencv-studio/shared';
import { formatMathText } from '../utils/mathFormatter';

interface ModelInspectorModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  selectedModelId?: string;
}

export const ModelInspectorModal: React.FC<ModelInspectorModalProps> = ({
  open,
  onOpenChange,
  selectedModelId = 'unet_custom',
}) => {
  const [activeModelId, setActiveModelId] = React.useState<string>(selectedModelId);

  React.useEffect(() => {
    if (selectedModelId) {
      setActiveModelId(selectedModelId);
    }
  }, [selectedModelId]);

  const activeModel: DetailedModelSpec =
    MODEL_METADATA_REGISTRY.find((m) => m.id === activeModelId) || MODEL_METADATA_REGISTRY[0];

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/70 backdrop-blur-md z-50 animate-fade-in" />
        <Dialog.Content className="fixed top-[50%] left-[50%] translate-x-[-50%] translate-y-[-50%] w-[90vw] max-w-3xl max-h-[85vh] overflow-y-auto glass-panel p-6 rounded-2xl border border-slate-700/80 shadow-2xl z-50 space-y-6 focus:outline-none">
          {/* Dialog Header */}
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-xl bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center">
                <Info className="w-5 h-5 text-cyan-400" />
              </div>
              <div>
                <Dialog.Title className="text-xl font-bold text-white">
                  Model Specs & Parameter Guide
                </Dialog.Title>
                <Dialog.Description className="text-xs text-slate-400">
                  Detailed neural architecture specs, weight files, and parameter tuning guide
                </Dialog.Description>
              </div>
            </div>

            <Dialog.Close className="w-8 h-8 rounded-xl bg-slate-900 border border-slate-800 hover:border-slate-700 flex items-center justify-center text-slate-400 hover:text-white transition">
              <X className="w-4 h-4" />
            </Dialog.Close>
          </div>

          {/* Model Selector Tabs */}
          <div className="flex space-x-2 overflow-x-auto pb-2 border-b border-slate-800/80">
            {MODEL_METADATA_REGISTRY.map((model) => (
              <button
                key={model.id}
                onClick={() => setActiveModelId(model.id)}
                className={`px-3 py-1.5 rounded-xl text-xs font-medium whitespace-nowrap border transition ${
                  activeModelId === model.id
                    ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300'
                    : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:bg-slate-800'
                }`}
              >
                {model.name}
              </button>
            ))}
          </div>

          {/* Active Model Content Card */}
          <div className="space-y-6">
            <div className="flex items-start justify-between flex-wrap gap-4 p-4 bg-slate-900/80 rounded-xl border border-slate-800">
              <div>
                <h3 className="text-lg font-bold text-white">{activeModel.name}</h3>
                <p className="text-xs text-slate-400 mt-1 max-w-xl">{activeModel.description}</p>
              </div>

              <div className="flex items-center space-x-2">
                <span className="px-2.5 py-1 rounded-lg text-[10px] font-mono uppercase bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                  {activeModel.category}
                </span>
                <a
                  href={activeModel.downloadUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-xs font-mono text-slate-200 border border-slate-700 rounded-lg inline-flex items-center space-x-1.5 transition"
                >
                  <Download className="w-3.5 h-3.5 text-cyan-400" />
                  <span>{activeModel.weightsFile}</span>
                </a>
              </div>
            </div>

            {/* Grid Technical Specifications */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-800/80 space-y-1">
                <div className="text-[10px] uppercase font-mono text-slate-400">Backbone Architecture</div>
                <div className="text-xs font-semibold text-slate-200">{activeModel.architectureBackbone}</div>
              </div>

              <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-800/80 space-y-1">
                <div className="text-[10px] uppercase font-mono text-slate-400">Input Tensor Dimensions</div>
                <div className="text-xs font-mono font-semibold text-cyan-400">{activeModel.inputTensor}</div>
              </div>

              <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-800/80 space-y-1">
                <div className="text-[10px] uppercase font-mono text-slate-400">Weight Precision / Quant</div>
                <div className="text-xs font-mono font-semibold text-amber-400">{activeModel.quantization}</div>
              </div>
            </div>

            {/* Parameter Adjustment Guide */}
            <div className="space-y-3 p-5 bg-slate-950/80 rounded-xl border border-slate-800">
              <h4 className="text-sm font-bold text-white flex items-center space-x-2">
                <BookOpen className="w-4 h-4 text-cyan-400" />
                <span>Parameter Tuning Guide</span>
              </h4>

              <div className="space-y-2.5">
                {Object.entries(activeModel.parametersExplanation).map(([param, desc]) => (
                  <div key={param} className="p-3 bg-slate-900/80 rounded-lg border border-slate-800/60 space-y-0.5">
                    <div className="text-xs font-semibold text-cyan-300 font-mono">{param}</div>
                    <div className="text-xs text-slate-300">{formatMathText(desc)}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Mathematical Formula Section (if applicable) */}
            {activeModel.mathOrFormula && (
              <div className="p-4 bg-cyan-950/20 border border-cyan-500/20 rounded-xl space-y-2">
                <div className="text-xs font-semibold text-cyan-300 flex items-center space-x-2">
                  <CheckCircle2 className="w-4 h-4 text-cyan-400" />
                  <span>Mathematical Operator Formulation</span>
                </div>
                <div className="font-mono text-xs text-cyan-200 bg-slate-950/80 p-3 rounded-lg border border-slate-800 overflow-x-auto whitespace-pre-wrap leading-relaxed">
                  {formatMathText(activeModel.mathOrFormula)}
                </div>
              </div>
            )}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};
