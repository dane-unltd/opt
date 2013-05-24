package linesearch

type Siso func(float64) float64

type Solver interface {
	Solve(obj, grad Siso, f0, g0, x1 float64) (float64, float64)
}
