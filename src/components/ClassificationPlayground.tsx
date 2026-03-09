import { useState, useEffect, useCallback, useMemo } from 'react';
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
import { RefreshCw, MousePointer2, Trash2 } from 'lucide-react';
import {
  DataPoint,
  DistanceMetric,
  WeightingScheme,
  getDistanceFunction,
  generateDecisionBoundaryGrid,
  generateTwoMoons,
  generateConcentricCircles,
} from '@/lib/knnEngine';

type DatasetType = 'two_moons' | 'circles' | 'custom';

interface PlotClickEvent {
  points: Array<{
    x?: number;
    y?: number;
    pointIndex?: number;
  }>;
}

/**
 * Classification Playground - Interactive KNN visualization
 * Features decision boundary contour plot with real-time updates
 */
export function ClassificationPlayground() {
  const [k, setK] = useState(5);
  const [distanceMetric, setDistanceMetric] = useState<DistanceMetric>('euclidean');
  const [minkowskiP, setMinkowskiP] = useState(2);
  const [weighting, setWeighting] = useState<WeightingScheme>('uniform');
  const [datasetType, setDatasetType] = useState<DatasetType>('two_moons');
  const [trainingData, setTrainingData] = useState<DataPoint[]>([]);
  const [customPoints, setCustomPoints] = useState<DataPoint[]>([]);
  const [selectedClass, setSelectedClass] = useState(0);
  const [resolution, setResolution] = useState(40);
  const [isCalculating, setIsCalculating] = useState(false);

  // Generate initial dataset
  useEffect(() => {
    regenerateDataset();
  }, [datasetType]);

  const regenerateDataset = useCallback(() => {
    let data: DataPoint[] = [];
    switch (datasetType) {
      case 'two_moons':
        data = generateTwoMoons(100, 0.15);
        break;
      case 'circles':
        data = generateConcentricCircles(100, 0.15);
        break;
      case 'custom':
        data = customPoints;
        break;
    }
    setTrainingData(data);
  }, [datasetType, customPoints]);

  // Distance function
  const distanceFn = useMemo(
    () => getDistanceFunction(distanceMetric, minkowskiP),
    [distanceMetric, minkowskiP]
  );

  // Calculate data bounds
  const bounds = useMemo(() => {
    if (trainingData.length === 0) {
      return { xMin: -2, xMax: 2, yMin: -2, yMax: 2 };
    }
    const xs = trainingData.map((p) => p.features[0]);
    const ys = trainingData.map((p) => p.features[1]);
    const padding = 0.5;
    return {
      xMin: Math.min(...xs) - padding,
      xMax: Math.max(...xs) + padding,
      yMin: Math.min(...ys) - padding,
      yMax: Math.max(...ys) + padding,
    };
  }, [trainingData]);

  // Generate decision boundary grid (memoized for performance)
  const gridData = useMemo(() => {
    if (trainingData.length < 2) return null;
    setIsCalculating(true);
    
    const result = generateDecisionBoundaryGrid(
      trainingData,
      Math.min(k, trainingData.length),
      distanceFn,
      weighting,
      [bounds.xMin, bounds.xMax],
      [bounds.yMin, bounds.yMax],
      resolution
    );
    
    setIsCalculating(false);
    return result;
  }, [trainingData, k, distanceFn, weighting, bounds, resolution]);

  // Handle click on plot to add custom point
  const handlePlotClick = useCallback(
    (event: PlotClickEvent) => {
      if (datasetType !== 'custom') return;
      
      const point = event.points[0];
      if (point && point.x !== undefined && point.y !== undefined) {
        const newPoint: DataPoint = {
          features: [point.x as number, point.y as number],
          label: selectedClass,
        };
        const newCustomPoints = [...customPoints, newPoint];
        setCustomPoints(newCustomPoints);
        setTrainingData(newCustomPoints);
      }
    },
    [datasetType, selectedClass, customPoints]
  );

  // Separate data by class
  const class0Points = trainingData.filter((p) => p.label === 0);
  const class1Points = trainingData.filter((p) => p.label === 1);

  // Plotly traces
  const traces: object[] = [];

  // Decision boundary contour
  if (gridData) {
    traces.push({
      type: 'contour',
      x: gridData.x,
      y: gridData.y,
      z: gridData.z,
      colorscale: [
        [0, 'rgba(59, 130, 246, 0.3)'],
        [1, 'rgba(249, 115, 22, 0.3)'],
      ],
      showscale: false,
      contours: {
        coloring: 'heatmap',
      },
      hoverinfo: 'skip',
    });
  }

  // Class 0 scatter points
  traces.push({
    type: 'scatter',
    mode: 'markers',
    x: class0Points.map((p) => p.features[0]),
    y: class0Points.map((p) => p.features[1]),
    marker: {
      size: 10,
      color: 'hsl(217, 91%, 60%)',
      line: { width: 2, color: 'hsl(217, 91%, 40%)' },
    },
    name: 'Class 0',
    hovertemplate: 'x: %{x:.2f}<br>y: %{y:.2f}<extra>Class 0</extra>',
  });

  // Class 1 scatter points
  traces.push({
    type: 'scatter',
    mode: 'markers',
    x: class1Points.map((p) => p.features[0]),
    y: class1Points.map((p) => p.features[1]),
    marker: {
      size: 10,
      color: 'hsl(32, 95%, 55%)',
      line: { width: 2, color: 'hsl(32, 95%, 40%)' },
    },
    name: 'Class 1',
    hovertemplate: 'x: %{x:.2f}<br>y: %{y:.2f}<extra>Class 1</extra>',
  });

  const layout = {
    autosize: true,
    height: 500,
    margin: { t: 20, r: 20, b: 40, l: 40 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(15, 23, 42, 0.5)',
    font: { color: '#94A3B8' },
    xaxis: {
      gridcolor: 'rgba(148, 163, 184, 0.1)',
      zerolinecolor: 'rgba(148, 163, 184, 0.2)',
      range: [bounds.xMin, bounds.xMax],
    },
    yaxis: {
      gridcolor: 'rgba(148, 163, 184, 0.1)',
      zerolinecolor: 'rgba(148, 163, 184, 0.2)',
      range: [bounds.yMin, bounds.yMax],
      scaleanchor: 'x',
    },
    legend: {
      x: 1,
      xanchor: 'right',
      y: 1,
      bgcolor: 'rgba(15, 23, 42, 0.8)',
    },
    dragmode: 'pan',
  };

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="section-header flex items-center justify-between">
          <span>Classification Mode</span>
          {isCalculating && (
            <Badge variant="outline" className="animate-pulse">
              Calculating...
            </Badge>
          )}
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
              max={Math.max(trainingData.length, 20)}
              step={1}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              {k === 1 ? 'Overfit (complex boundary)' : k > 15 ? 'Underfit (smooth)' : 'Balanced'}
            </p>
          </div>

          {/* Dataset Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Dataset</label>
            <Select value={datasetType} onValueChange={(v) => setDatasetType(v as DatasetType)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="two_moons">Two Moons</SelectItem>
                <SelectItem value="circles">Concentric Circles</SelectItem>
                <SelectItem value="custom">Custom (Click to add)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Distance Metric */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Distance Metric</label>
            <Select value={distanceMetric} onValueChange={(v) => setDistanceMetric(v as DistanceMetric)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="euclidean">Euclidean (L2)</SelectItem>
                <SelectItem value="manhattan">Manhattan (L1)</SelectItem>
                <SelectItem value="minkowski">Minkowski</SelectItem>
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
        </div>

        {/* Minkowski P slider (when selected) */}
        {distanceMetric === 'minkowski' && (
          <div className="flex items-center gap-4 p-3 bg-muted/20 rounded-lg">
            <label className="text-sm font-medium whitespace-nowrap">
              Minkowski p = {minkowskiP.toFixed(1)}
            </label>
            <Slider
              value={[minkowskiP]}
              onValueChange={(v) => setMinkowskiP(v[0])}
              min={0.5}
              max={5}
              step={0.1}
              className="flex-1"
            />
          </div>
        )}

        {/* Custom mode controls */}
        {datasetType === 'custom' && (
          <div className="flex items-center gap-4 p-3 bg-primary/10 rounded-lg border border-primary/30">
            <MousePointer2 className="w-5 h-5 text-primary" />
            <span className="text-sm">Click on the plot to add points</span>
            <div className="flex gap-2 ml-auto">
              <Button
                size="sm"
                variant={selectedClass === 0 ? 'default' : 'outline'}
                onClick={() => setSelectedClass(0)}
              >
                Class 0
              </Button>
              <Button
                size="sm"
                variant={selectedClass === 1 ? 'default' : 'outline'}
                onClick={() => setSelectedClass(1)}
              >
                Class 1
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  setCustomPoints([]);
                  setTrainingData([]);
                }}
              >
                <Trash2 className="w-4 h-4" />
              </Button>
            </div>
          </div>
        )}

        {/* Resolution control */}
        <div className="flex items-center gap-4">
          <label className="text-sm text-muted-foreground">
            Grid Resolution: {resolution}
          </label>
          <Slider
            value={[resolution]}
            onValueChange={(v) => setResolution(v[0])}
            min={20}
            max={80}
            step={5}
            className="w-32"
          />
          <Button
            size="sm"
            variant="outline"
            onClick={regenerateDataset}
            disabled={datasetType === 'custom'}
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Regenerate
          </Button>
        </div>

        {/* Plotly Chart */}
        <div className="plotly-container">
          <Plot
            data={traces as any}
            layout={layout as any}
            config={{
              responsive: true,
              displayModeBar: true,
              modeBarButtonsToRemove: ['lasso2d', 'select2d'] as any,
            }}
            onClick={handlePlotClick as any}
            style={{ width: '100%' }}
          />
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="p-3 bg-muted/20 rounded-lg">
            <p className="text-2xl font-bold text-primary">{trainingData.length}</p>
            <p className="text-xs text-muted-foreground">Total Points</p>
          </div>
          <div className="p-3 bg-muted/20 rounded-lg">
            <p className="text-2xl font-bold text-class-0">{class0Points.length}</p>
            <p className="text-xs text-muted-foreground">Class 0</p>
          </div>
          <div className="p-3 bg-muted/20 rounded-lg">
            <p className="text-2xl font-bold text-class-1">{class1Points.length}</p>
            <p className="text-xs text-muted-foreground">Class 1</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
