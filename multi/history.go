package multi

import (
	"time"
)

type History struct {
	T    []time.Duration
	X    [][]float64
	Obj  []float64
	Grad [][]float64
}

func (h *History) Update(sol *Solution, stats *Stats) Status {
	if h.T != nil {
		h.T = append(h.T, stats.Time)
	}
	if h.X != nil {
		xt := make([]float64, len(sol.X))
		copy(xt, sol.X)
		h.X = append(h.X, xt)
	}
	if h.Grad != nil {
		if sol.Grad != nil {
			gt := make([]float64, len(sol.Grad))
			copy(gt, sol.Grad)
			h.Grad = append(h.Grad, gt)
		}
	}
	if h.Obj != nil {
		h.Obj = append(h.Obj, sol.Obj)
	}
	return NotTerminated
}
