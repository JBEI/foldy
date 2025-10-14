import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { PlotContainer } from '../../util/plotComponents';

interface FewShotDebugPlotsProps {
    debugData: any;
}

const createPlotData = (debugData: any) => {
    if (!debugData || !debugData.pretrain_metrics || !debugData.finetune_metrics) {
        return {
            pretrain: [],
            finetune: []
        };
    }

    // Create pretrain traces - one for each model's train and val loss
    const pretrainData: any[] = [];
    const finetuneData: any[] = [];

    // Get frequency data for epoch conversion
    const pretrainValFreq = debugData.pretrain_val_frequency || 1;
    const finetuneValFreq = debugData.finetune_val_frequency || 1;

    // Inside the pretrain metrics loop:
    debugData.pretrain_metrics.forEach((model: any, index: number) => {
        if (model.train_loss && model.train_loss.length > 0) {
            // Convert iterations to epochs: epoch = iteration * val_frequency
            const trainEpochs = Array.from({ length: model.train_loss.length }, (_, i) => (i + 1) * pretrainValFreq);
            // Filter out NaN values
            const cleanTrainLoss = model.train_loss.map((val: number) => (isNaN(val) || val === null) ? null : val);

            pretrainData.push({
                x: trainEpochs,
                y: cleanTrainLoss,
                name: `Model ${index + 1} Train`,
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#1890ff', // Ant Design blue for train
                    width: 2,
                    dash: 'solid'
                }
            });
        }

        if (model.val_loss && model.val_loss.length > 0) {
            // Convert iterations to epochs: epoch = iteration * val_frequency
            const valEpochs = Array.from({ length: model.val_loss.length }, (_, i) => (i + 1) * pretrainValFreq);
            // Filter out NaN values and find minimum for highlighting
            const cleanValLoss = model.val_loss.map((val: number) => (isNaN(val) || val === null) ? null : val);

            pretrainData.push({
                x: valEpochs,
                y: cleanValLoss,
                name: `Model ${index + 1} Val`,
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#fa541c', // Ant Design orange for validation
                    width: 1,
                    dash: 'solid',
                    opacity: 0.5,
                }
            });

            // Add scatter point for minimum validation loss
            const validLosses = model.val_loss.filter((val: number) => !isNaN(val) && val !== null);
            if (validLosses.length > 0) {
                const minVal = Math.min(...validLosses);
                const minIdx = model.val_loss.findIndex((val: number) => val === minVal);
                if (minIdx !== -1) {
                    pretrainData.push({
                        x: [(minIdx + 1) * pretrainValFreq],
                        y: [minVal],
                        name: `Model ${index + 1} Min Val`,
                        type: 'scatter',
                        mode: 'markers',
                        marker: {
                            color: '#fa541c',
                            size: 8,
                            symbol: 'star'
                        },
                        showlegend: false
                    });
                }
            }
        }
    });

    // Inside the finetune metrics loop:
    debugData.finetune_metrics.forEach((model: any, index: number) => {
        if (model.train_loss && model.train_loss.some((val: number) => val !== 0 && val !== null && !isNaN(val))) {
            // Convert iterations to epochs: epoch = iteration * val_frequency
            const trainEpochs = Array.from({ length: model.train_loss.length }, (_, i) => (i + 1) * finetuneValFreq);
            // Filter out zeros and NaN values which appear to be placeholders
            const cleanTrainLoss = model.train_loss.map((val: number) => (val === 0 || isNaN(val) || val === null) ? null : val);

            finetuneData.push({
                x: trainEpochs,
                y: cleanTrainLoss,
                name: `Model ${index + 1} Train`,
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#1890ff', // Ant Design blue for train
                    width: 1,
                    dash: 'solid',
                    opacity: 0.5,
                }
            });
        }

        if (model.val_loss && model.val_loss.some((val: number) => val !== 0 && val !== null && !isNaN(val))) {
            // Convert iterations to epochs: epoch = iteration * val_frequency
            const valEpochs = Array.from({ length: model.val_loss.length }, (_, i) => (i + 1) * finetuneValFreq);
            // Filter out zeros and NaN values which appear to be placeholders
            const cleanValLoss = model.val_loss.map((val: number) => (val === 0 || isNaN(val) || val === null) ? null : val);

            finetuneData.push({
                x: valEpochs,
                y: cleanValLoss,
                name: `Model ${index + 1} Val`,
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#fa541c', // Ant Design orange for validation
                    width: 1,
                    dash: 'solid',
                    opacity: 0.5,
                }
            });

            // Add scatter point for minimum validation loss
            const validLosses = model.val_loss.filter((val: number) => val !== 0 && !isNaN(val) && val !== null);
            if (validLosses.length > 0) {
                const minVal = Math.min(...validLosses);
                const minIdx = model.val_loss.findIndex((val: number) => val === minVal);
                if (minIdx !== -1) {
                    finetuneData.push({
                        x: [(minIdx + 1) * finetuneValFreq],
                        y: [minVal],
                        name: `Model ${index + 1} Min Val`,
                        type: 'scatter',
                        mode: 'markers',
                        marker: {
                            color: '#fa541c',
                            size: 8,
                            symbol: 'star'
                        },
                        showlegend: false
                    });
                }
            }
        }
    });

    return {
        pretrain: pretrainData,
        finetune: finetuneData
    };
};

// Create a memoized component to render the plotly charts
const FewShotDebugPlots: React.FC<FewShotDebugPlotsProps> = React.memo(({ debugData }) => {
    const plotData = useMemo(() => createPlotData(debugData), [debugData]);

    const pretrainPlot = useMemo(() => (
        <Plot
            data={plotData.pretrain}
            layout={{
                title: {
                    text: 'Warm Start Loss',
                    font: { size: 14, color: '#262626' }
                },
                autosize: true,
                margin: { l: 50, r: 20, t: 40, b: 80 },
                xaxis: {
                    title: { text: 'Epoch', font: { size: 12, color: '#595959' } },
                    tickfont: { size: 10, color: '#8c8c8c' },
                    gridcolor: '#f0f0f0',
                    linecolor: '#d9d9d9'
                },
                yaxis: {
                    title: { text: 'Loss', font: { size: 12, color: '#595959' } },
                    tickfont: { size: 10, color: '#8c8c8c' },
                    gridcolor: '#f0f0f0',
                    linecolor: '#d9d9d9'
                },
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif' },
                legend: {
                    orientation: 'h',
                    x: 0.5,
                    xanchor: 'center',
                    y: -0.15,
                    yanchor: 'top',
                    font: { size: 10, color: '#595959' }
                }
            }}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler={true}
            config={{
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            }}
        />
    ), [plotData.pretrain]);

    const finetunePlot = useMemo(() => (
        <Plot
            data={plotData.finetune}
            layout={{
                title: {
                    text: 'Finetuning Loss',
                    font: { size: 14, color: '#262626' }
                },
                autosize: true,
                margin: { l: 50, r: 20, t: 40, b: 80 },
                xaxis: {
                    title: { text: 'Epoch', font: { size: 12, color: '#595959' } },
                    tickfont: { size: 10, color: '#8c8c8c' },
                    gridcolor: '#f0f0f0',
                    linecolor: '#d9d9d9'
                },
                yaxis: {
                    title: { text: 'Loss', font: { size: 12, color: '#595959' } },
                    tickfont: { size: 10, color: '#8c8c8c' },
                    gridcolor: '#f0f0f0',
                    linecolor: '#d9d9d9'
                },
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif' },
                legend: {
                    orientation: 'h',
                    x: 0.5,
                    xanchor: 'center',
                    y: -0.15,
                    yanchor: 'top',
                    font: { size: 10, color: '#595959' }
                }
            }}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler={true}
            config={{
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            }}
        />
    ), [plotData.finetune]);

    if (!debugData) return null;

    return (
        <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '20px',
            height: '320px'
        }}>
            {/* Pretraining loss plot */}
            <div style={{
                height: '100%',
                padding: '8px',
                borderRadius: '6px',
                backgroundColor: '#fafafa'
            }}>
                {pretrainPlot}
            </div>

            {/* Finetuning loss plot */}
            <div style={{
                height: '100%',
                padding: '8px',
                borderRadius: '6px',
                backgroundColor: '#fafafa'
            }}>
                {finetunePlot}
            </div>
        </div>
    );
});

FewShotDebugPlots.displayName = 'FewShotDebugPlots';

export default FewShotDebugPlots;
