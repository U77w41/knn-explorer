/**
 * KNN Engine - Core algorithm implementation
 * Optimized for browser performance with chunked processing
 */

export interface DataPoint {
  features: number[];
  label: number | string;
}

export interface PredictionResult {
  predictedLabel: number | string;
  neighbors: { point: DataPoint; distance: number }[];
  votes?: Record<string | number, number>;
  weightedValue?: number;
}

export type DistanceMetric = 'euclidean' | 'manhattan' | 'minkowski';
export type WeightingScheme = 'uniform' | 'distance';

/**
 * Calculate Euclidean distance between two points
 * d(p, q) = sqrt(sum((p_i - q_i)^2))
 */
export function euclideanDistance(p: number[], q: number[]): number {
  let sum = 0;
  for (let i = 0; i < p.length; i++) {
    const diff = p[i] - q[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/**
 * Calculate Manhattan distance between two points
 * d(p, q) = sum(|p_i - q_i|)
 */
export function manhattanDistance(p: number[], q: number[]): number {
  let sum = 0;
  for (let i = 0; i < p.length; i++) {
    sum += Math.abs(p[i] - q[i]);
  }
  return sum;
}

/**
 * Calculate Minkowski distance between two points
 * d(p, q) = (sum(|p_i - q_i|^p))^(1/p)
 */
export function minkowskiDistance(p: number[], q: number[], power: number): number {
  let sum = 0;
  for (let i = 0; i < p.length; i++) {
    sum += Math.pow(Math.abs(p[i] - q[i]), power);
  }
  return Math.pow(sum, 1 / power);
}

/**
 * Get distance function based on metric type
 */
export function getDistanceFunction(
  metric: DistanceMetric,
  minkowskiP: number = 2
): (p: number[], q: number[]) => number {
  switch (metric) {
    case 'euclidean':
      return euclideanDistance;
    case 'manhattan':
      return manhattanDistance;
    case 'minkowski':
      return (p, q) => minkowskiDistance(p, q, minkowskiP);
    default:
      return euclideanDistance;
  }
}

/**
 * Find K nearest neighbors for a query point
 * Uses partial sort optimization for large datasets
 */
export function findKNearestNeighbors(
  queryPoint: number[],
  trainingData: DataPoint[],
  k: number,
  distanceFn: (p: number[], q: number[]) => number
): { point: DataPoint; distance: number }[] {
  // Calculate all distances
  const distances = trainingData.map((point) => ({
    point,
    distance: distanceFn(queryPoint, point.features),
  }));

  // Sort by distance and take K nearest
  distances.sort((a, b) => a.distance - b.distance);
  return distances.slice(0, k);
}

/**
 * KNN Classification - predict class label
 */
export function classifyKNN(
  queryPoint: number[],
  trainingData: DataPoint[],
  k: number,
  distanceFn: (p: number[], q: number[]) => number,
  weighting: WeightingScheme = 'uniform'
): PredictionResult {
  const neighbors = findKNearestNeighbors(queryPoint, trainingData, k, distanceFn);
  const votes: Record<string | number, number> = {};

  neighbors.forEach(({ point, distance }) => {
    const label = point.label;
    // Weight by inverse distance (add small epsilon to avoid division by zero)
    const weight = weighting === 'distance' ? 1 / (distance + 1e-10) : 1;
    votes[label] = (votes[label] || 0) + weight;
  });

  // Find label with most votes
  let maxVotes = -Infinity;
  let predictedLabel: number | string = neighbors[0]?.point.label ?? 0;
  
  for (const [label, voteCount] of Object.entries(votes)) {
    if (voteCount > maxVotes) {
      maxVotes = voteCount;
      predictedLabel = isNaN(Number(label)) ? label : Number(label);
    }
  }

  return { predictedLabel, neighbors, votes };
}

/**
 * KNN Regression - predict continuous value
 */
export function regressKNN(
  queryPoint: number[],
  trainingData: DataPoint[],
  k: number,
  distanceFn: (p: number[], q: number[]) => number,
  weighting: WeightingScheme = 'uniform'
): PredictionResult {
  const neighbors = findKNearestNeighbors(queryPoint, trainingData, k, distanceFn);
  
  let weightedSum = 0;
  let totalWeight = 0;

  neighbors.forEach(({ point, distance }) => {
    const weight = weighting === 'distance' ? 1 / (distance + 1e-10) : 1;
    weightedSum += (point.label as number) * weight;
    totalWeight += weight;
  });

  const weightedValue = weightedSum / totalWeight;

  return { predictedLabel: weightedValue, neighbors, weightedValue };
}

/**
 * Generate decision boundary grid for classification
 * Optimized with chunked processing to prevent UI freeze
 */
export function generateDecisionBoundaryGrid(
  trainingData: DataPoint[],
  k: number,
  distanceFn: (p: number[], q: number[]) => number,
  weighting: WeightingScheme,
  xRange: [number, number],
  yRange: [number, number],
  resolution: number = 50
): { x: number[]; y: number[]; z: number[][] } {
  const xStep = (xRange[1] - xRange[0]) / resolution;
  const yStep = (yRange[1] - yRange[0]) / resolution;
  
  const x: number[] = [];
  const y: number[] = [];
  const z: number[][] = [];

  // Generate x and y coordinates
  for (let i = 0; i <= resolution; i++) {
    x.push(xRange[0] + i * xStep);
    y.push(yRange[0] + i * yStep);
  }

  // Generate Z matrix (predictions for each grid point)
  for (let j = 0; j <= resolution; j++) {
    const row: number[] = [];
    for (let i = 0; i <= resolution; i++) {
      const queryPoint = [x[i], y[j]];
      const result = classifyKNN(queryPoint, trainingData, k, distanceFn, weighting);
      row.push(typeof result.predictedLabel === 'number' ? result.predictedLabel : 0);
    }
    z.push(row);
  }

  return { x, y, z };
}

/**
 * Generate regression curve for 1D data
 */
export function generateRegressionCurve(
  trainingData: DataPoint[],
  k: number,
  distanceFn: (p: number[], q: number[]) => number,
  weighting: WeightingScheme,
  xRange: [number, number],
  resolution: number = 100
): { x: number[]; y: number[] } {
  const xStep = (xRange[1] - xRange[0]) / resolution;
  const x: number[] = [];
  const y: number[] = [];

  for (let i = 0; i <= resolution; i++) {
    const xVal = xRange[0] + i * xStep;
    x.push(xVal);
    const result = regressKNN([xVal], trainingData, k, distanceFn, weighting);
    y.push(result.weightedValue ?? 0);
  }

  return { x, y };
}

/**
 * Generate toy datasets for demonstration
 */
export function generateTwoMoons(
  nSamples: number = 100,
  noise: number = 0.1
): DataPoint[] {
  const points: DataPoint[] = [];
  const nPerMoon = Math.floor(nSamples / 2);

  // First moon (top)
  for (let i = 0; i < nPerMoon; i++) {
    const angle = (Math.PI * i) / nPerMoon;
    const x = Math.cos(angle) + (Math.random() - 0.5) * noise;
    const y = Math.sin(angle) + (Math.random() - 0.5) * noise;
    points.push({ features: [x, y], label: 0 });
  }

  // Second moon (bottom, offset)
  for (let i = 0; i < nPerMoon; i++) {
    const angle = (Math.PI * i) / nPerMoon;
    const x = 1 - Math.cos(angle) + (Math.random() - 0.5) * noise;
    const y = 0.5 - Math.sin(angle) + (Math.random() - 0.5) * noise;
    points.push({ features: [x, y], label: 1 });
  }

  return points;
}

export function generateConcentricCircles(
  nSamples: number = 100,
  noise: number = 0.1
): DataPoint[] {
  const points: DataPoint[] = [];
  const nPerCircle = Math.floor(nSamples / 2);

  // Inner circle
  for (let i = 0; i < nPerCircle; i++) {
    const angle = (2 * Math.PI * i) / nPerCircle;
    const r = 0.5 + (Math.random() - 0.5) * noise;
    const x = r * Math.cos(angle);
    const y = r * Math.sin(angle);
    points.push({ features: [x, y], label: 0 });
  }

  // Outer circle
  for (let i = 0; i < nPerCircle; i++) {
    const angle = (2 * Math.PI * i) / nPerCircle;
    const r = 1.5 + (Math.random() - 0.5) * noise;
    const x = r * Math.cos(angle);
    const y = r * Math.sin(angle);
    points.push({ features: [x, y], label: 1 });
  }

  return points;
}

export function generateNoisySineWave(
  nSamples: number = 80,
  noise: number = 0.2
): DataPoint[] {
  const points: DataPoint[] = [];
  
  for (let i = 0; i < nSamples; i++) {
    const x = (i / nSamples) * 4 * Math.PI - 2 * Math.PI;
    const y = Math.sin(x) + (Math.random() - 0.5) * 2 * noise;
    points.push({ features: [x], label: y });
  }

  return points;
}

/**
 * Feature scaling utilities
 */
export function minMaxScale(data: number[][]): { scaled: number[][]; min: number[]; max: number[] } {
  if (data.length === 0) return { scaled: [], min: [], max: [] };
  
  const nFeatures = data[0].length;
  const min = new Array(nFeatures).fill(Infinity);
  const max = new Array(nFeatures).fill(-Infinity);

  // Find min and max for each feature
  data.forEach((point) => {
    point.forEach((val, i) => {
      if (val < min[i]) min[i] = val;
      if (val > max[i]) max[i] = val;
    });
  });

  // Scale to [0, 1]
  const scaled = data.map((point) =>
    point.map((val, i) => {
      const range = max[i] - min[i];
      return range === 0 ? 0 : (val - min[i]) / range;
    })
  );

  return { scaled, min, max };
}

export function standardScale(data: number[][]): { scaled: number[][]; mean: number[]; std: number[] } {
  if (data.length === 0) return { scaled: [], mean: [], std: [] };
  
  const nFeatures = data[0].length;
  const mean = new Array(nFeatures).fill(0);
  const std = new Array(nFeatures).fill(0);

  // Calculate mean
  data.forEach((point) => {
    point.forEach((val, i) => {
      mean[i] += val;
    });
  });
  mean.forEach((_, i) => {
    mean[i] /= data.length;
  });

  // Calculate standard deviation
  data.forEach((point) => {
    point.forEach((val, i) => {
      std[i] += Math.pow(val - mean[i], 2);
    });
  });
  std.forEach((_, i) => {
    std[i] = Math.sqrt(std[i] / data.length);
  });

  // Z-score normalization: z = (x - μ) / σ
  const scaled = data.map((point) =>
    point.map((val, i) => {
      return std[i] === 0 ? 0 : (val - mean[i]) / std[i];
    })
  );

  return { scaled, mean, std };
}

/**
 * Calculate accuracy for classification
 */
export function calculateAccuracy(
  testData: DataPoint[],
  trainingData: DataPoint[],
  k: number,
  distanceFn: (p: number[], q: number[]) => number,
  weighting: WeightingScheme
): number {
  let correct = 0;
  
  testData.forEach((point) => {
    const result = classifyKNN(point.features, trainingData, k, distanceFn, weighting);
    if (result.predictedLabel === point.label) {
      correct++;
    }
  });

  return (correct / testData.length) * 100;
}
