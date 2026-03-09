import { useState, useEffect, useMemo, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { RefreshCw, TrendingUp } from 'lucide-react';
import {
  DataPoint,
  DistanceMetric,
  WeightingScheme,
  getDistanceFunction,
  generateRegressionCurve,
  generateNoisySineWave,
} from '@/lib/knnEngine';

/**
 * Regression Playground - Interactive KNN regression visualization
 * Shows how K affects the smoothness of the regression curve
 */
export function RegressionPlayground() {
  const [k, setK] = useState(5);
  const [distanceMetric, setDistanceMetric] = useState<DistanceMetric>('euclidean');
  const [weighting, setWeighting] = useState<WeightingScheme>('uniform');
  const [trainingData, setTrainingData] = useState<DataPoint[]>([]);
  const [noise, setNoise] = useState(0.3);
  const [nSamples, setNSamples] = useState(60);

  // Generate initial dataset
  useEffect(() => {
    regenerateDataset();
  }, [noise, nSamples]);

  const regenerateDataset = useCallback(() => {
    setTrainingData(generateNoisySineWave(nSamples, noise));
  }, [noise, nSamples]);

  // Distance function
  const distanceFn = useMemo(
    () => getDistanceFunction(distanceMetric),
    [distanceMetric]
  );

  // Calculate data bounds
  const bounds = useMemo(() => {
    if (trainingData.length === 0) {
      return { xMin: -Math.PI * 2, xMax: Math.PI * 2, yMin: -2, yMax: 2 };
    }
    const xs = trainingData.map((p) => p.features[0]);
    const ys = trainingData.map((p) => p.label as number);
    return {
      xMin: Math.min(...xs) - 0.5,
      xMax: Math.max(...xs) + 0.5,
      yMin: Math.min(...ys) - 0.5,
      yMax: Math.max(...ys) + 0.5,
    };
  }, [trainingData]);

  // Generate regression curve
  const regressionCurve = useMemo(() => {
    if (trainingData.length < 2) return null;
    return generateRegressionCurve(
      trainingData,
      Math.min(k, trainingData.length),
      distanceFn,
      weighting,
      [bounds.xMin, bounds.xMax],
      200
    );
  }, [trainingData, k, distanceFn, weighting, bounds]);

  // True sine wave for comparison
  const trueSine = useMemo(() => {
    const x: number[] = [];
    const y: number[] = [];
    for (let i = 0; i <= 200; i++) {
      const xVal = bounds.xMin + (i / 200) * (bounds.xMax - bounds.xMin);
      x.push(xVal);
      y.push(Math.sin(xVal));
    }
    return { x, y };
  }, [bounds]);

  // Calculate MSE
  const mse = useMemo(() => {
    if (!regressionCurve || trainingData.length === 0) return 0;
    let sumSquaredError = 0;
    trainingData.forEach((point) => {
      const trueValue = Math.sin(point.features[0]);
      sumSquaredError += Math.pow((point.label as number) - trueValue, 2);
    });
    return sumSquaredError / trainingData.length;
  }, [trainingData]);

  // Plotly traces
  const traces: Plotly.Data[] = [
    // True sine wave (dashed line)
    {
      type: 'scatter',
      mode: 'lines',
      x: trueSine.x,
      y: trueSine.y,
      line: { color: '#22C55E', width: 2, dash: 'dash' },
      name: 'True Function (sin)',
      hovertemplate: 'x: %{x:.2f}<br>sin(x): %{y:.2f}<extra></extra>',
    },
    // Training data points
    {
      type: 'scatter',
      mode: 'markers',
      x: trainingData.map((p) => p.features[0]),
      y: trainingData.map((p) => p.label as number),
      marker: {
        size: 8,
        color: '#3B82F6',
        opacity: 0.7,
        line: { width: 1, color: '#1E40AF' },
      },
      name: 'Training Data',
      hovertemplate: 'x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
    },
  ];

  // KNN regression curve
  if (regressionCurve) {
    traces.push({
      type: 'scatter',
      mode: 'lines',
      x: regressionCurve.x,
      y: regressionCurve.y,
      line: { color: '#F97316', width: 3 },
      name: `KNN Regression (K=${k})`,
      hovertemplate: 'x: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>',
    });
  }

  const layout: Partial<Plotly.Layout> = {
    autosize: true,
    height: 450,
    margin: { t: 20, r: 20, b: 40, l: 50 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(15, 23, 42, 0.5)',
    font: { color: '#94A3B8' },
    xaxis: {
      gridcolor: 'rgba(148, 163, 184, 0.1)',
      zerolinecolor: 'rgba(148, 163, 184, 0.3)',
      title: { text: 'x', font: { size: 12 } },
    },
    yaxis: {
      gridcolor: 'rgba(148, 163, 184, 0.1)',
      zerolinecolor: 'rgba(148, 163, 184, 0.3)',
      title: { text: 'y', font: { size: 12 } },
      range: [-2, 2],
    },
    legend: {
      x: 0,
      y: 1,
      bgcolor: 'rgba(15, 23, 42, 0.8)',
    },
  };

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="section-header flex items-center gap-3">
          <TrendingUp className="w-6 h-6 text-secondary" />
          Regression Mode
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* K Slider */}
          <div className="space-y-2">
            <label className="text-sm font-medium flex items-center justify-between">
              <span>K (neighbors)</span>
              <Badge variant="secondary">{k}</Badge>
            </label>
            <Slider
              value={[k]}
              onValueChange={(v) => setK(v[0])}
              min={1}
              max={Math.min(trainingData.length, 30)}
              step={1}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              {k === 1 ? 'Step function (overfit)' : k > 15 ? 'Very smooth (underfit)' : 'Balanced'}
            </p>
          </div>

          {/* Distance Metric */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Distance Metric</label>
            <Select value={distanceMetric} onValueChange={(v) => setDistanceMetric(v as DistanceMetric)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="euclidean">Euclidean</SelectItem>
                <SelectItem value="manhattan">Manhattan</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Weighting */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Weighting</label>
            <Select value={weighting} onValueChange={(v) => setWeighting(v as WeightingScheme)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="uniform">Uniform</SelectItem>
                <SelectItem value="distance">Distance (1/d)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Regenerate */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Actions</label>
            <Button
              onClick={regenerateDataset}
              variant="outline"
              className="w-full"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              New Data
            </Button>
          </div>
        </div>

        {/* Data generation controls */}
        <div className="grid grid-cols-2 gap-4 p-4 bg-muted/20 rounded-lg">
          <div className="space-y-2">
            <label className="text-sm text-muted-foreground flex justify-between">
              <span>Noise Level</span>
              <span>{noise.toFixed(2)}</span>
            </label>
            <Slider
              value={[noise]}
              onValueChange={(v) => setNoise(v[0])}
              min={0}
              max={0.8}
              step={0.05}
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm text-muted-foreground flex justify-between">
              <span>Sample Count</span>
              <span>{nSamples}</span>
            </label>
            <Slider
              value={[nSamples]}
              onValueChange={(v) => setNSamples(v[0])}
              min={20}
              max={150}
              step={10}
            />
          </div>
        </div>

        {/* Plotly Chart */}
        <div className="plotly-container">
          <Plot
            data={traces}
            layout={layout}
            config={{
              responsive: true,
              displayModeBar: true,
              modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            }}
            style={{ width: '100%' }}
          />
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="p-3 bg-muted/20 rounded-lg">
            <p className="text-2xl font-bold text-primary">{trainingData.length}</p>
            <p className="text-xs text-muted-foreground">Data Points</p>
          </div>
          <div className="p-3 bg-muted/20 rounded-lg">
            <p className="text-2xl font-bold text-secondary">{k}</p>
            <p className="text-xs text-muted-foreground">Neighbors</p>
          </div>
          <div className="p-3 bg-muted/20 rounded-lg">
            <p className="text-2xl font-bold text-accent">{mse.toFixed(4)}</p>
            <p className="text-xs text-muted-foreground">MSE (noise)</p>
          </div>
        </div>

        {/* Explanation */}
        <div className="p-4 bg-secondary/10 rounded-lg border border-secondary/30">
          <h4 className="font-semibold text-secondary mb-2">How K affects regression:</h4>
          <ul className="text-sm text-muted-foreground space-y-1">
            <li>• <strong>K=1:</strong> Each prediction equals the nearest neighbor's value (step function)</li>
            <li>• <strong>Small K:</strong> Captures local patterns but sensitive to noise</li>
            <li>• <strong>Large K:</strong> Smoother predictions but may miss local variations</li>
            <li>• <strong>Distance weighting:</strong> Gives more influence to closer neighbors</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
