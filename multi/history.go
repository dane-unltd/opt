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

func (h *History) Update(m *Result) Status {
	if h.T != nil {
		h.T = append(h.T, m.Time)
	}
	if h.X != nil {
		xt := make([]float64, len(m.X))
		copy(xt, m.X)
		h.X = append(h.X, xt)
	}
	if h.Grad != nil {
		if m.Grad != nil {
			gt := make([]float64, len(m.Grad))
			copy(gt, m.Grad)
			h.Grad = append(h.Grad, gt)
		}
	}
	if h.Obj != nil {
		h.Obj = append(h.Obj, m.Obj)
	}
	return NotTerminated
}
