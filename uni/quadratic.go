package uni

import (
	"errors"
	"math"
	"time"
)

//Exact line search for strictly quasi-convex functions
type Quadratic struct {
	Tol     float64
	IterMax int
	TimeMax time.Duration
}

func NewQuadratic() *Quadratic {
	return &Quadratic{Tol: 1e-12, IterMax: 100, TimeMax: time.Second}
}

func (sol *Quadratic) Solve(m *Model) error {
	var err error = nil

	tStart := time.Now()

	if math.IsNaN(m.ObjLB) {
		m.ObjLB = m.Obj(m.LB)
	}

	fNew := 0.0
	xNew := 0.0

	m.Iter = 0

	if math.IsInf(m.UB, 1) {
		xNew = m.X
		fNew = m.ObjX
		if math.IsNaN(xNew) {
			xNew = 1
			fNew = m.Obj(xNew)
		}
		if math.IsNaN(fNew) {
			fNew = m.Obj(xNew)
		}

		if fNew < m.ObjLB {
			m.ObjX = fNew
			m.X = xNew

			m.UB = 2 * m.X
			m.ObjUB = m.Obj(m.UB)
			for ; m.Iter < sol.IterMax && m.ObjUB <= m.ObjX; m.Iter++ {
				m.UB *= 2
				m.ObjUB = m.Obj(m.UB)

				m.Time = time.Since(tStart)
				m.DoCallbacks()
				if m.Time > sol.TimeMax {
					err = errors.New("Time limit reached")
					goto done
				}
			}
		} else {
			m.ObjUB = fNew
			m.UB = xNew

			m.X = 0.5 * m.UB
			m.ObjX = m.Obj(m.X)
			for ; m.Iter < sol.IterMax && m.ObjX >= m.ObjLB; m.Iter++ {
				m.X *= 0.5
				m.ObjX = m.Obj(m.X)

				m.Time = time.Since(tStart)
				m.DoCallbacks()
				if m.Time > sol.TimeMax {
					err = errors.New("Time limit reached")
					goto done
				}
			}
		}
	} else {
		m.UB = m.UB
		m.ObjUB = m.ObjUB
		if math.IsNaN(m.ObjUB) {
			m.ObjUB = m.Obj(m.UB)
		}
		if m.ObjUB < m.ObjLB {
			m.X = m.UB - sol.Tol
			m.ObjX = m.Obj(m.X)
			if m.ObjX >= m.ObjUB {
				m.X = m.UB
				m.ObjX = m.ObjUB
				goto done
			}
		} else {
			m.X = 0.5 * m.UB
			m.ObjX = m.Obj(m.X)
			for ; m.Iter < sol.IterMax && m.ObjX >= m.ObjLB; m.Iter++ {
				m.X *= 0.5
				m.ObjX = m.Obj(m.X)

				m.Time = time.Since(tStart)
				m.DoCallbacks()
				if m.Time > sol.TimeMax {
					err = errors.New("Time limit reached")
					goto done
				}
			}
		}
	}

	for ; m.UB-m.LB > sol.Tol && m.Iter < sol.IterMax; m.Iter++ {
		xNew = -0.5 * (m.UB*m.UB*(m.ObjLB-m.ObjX) + m.X*m.X*(m.ObjUB-m.ObjLB) + m.LB*m.LB*(m.ObjX-m.ObjUB)) /
			(m.UB*(m.ObjX-m.ObjLB) + m.X*(m.ObjLB-m.ObjUB) + m.LB*(m.ObjUB-m.ObjX))
		fNew = m.Obj(xNew)
		if xNew > m.X {
			if fNew >= m.ObjX {
				m.UB = xNew
			} else {
				m.LB = m.X
				m.X = xNew
				m.ObjX = fNew
			}
		} else {
			if fNew >= m.ObjX {
				m.LB = xNew
			} else {
				m.UB = m.X
				m.X = xNew
				m.ObjX = fNew
			}
		}

		m.Time = time.Since(tStart)
		m.DoCallbacks()
		if m.Time > sol.TimeMax {
			err = errors.New("Time limit reached")
			goto done
		}
	}

done:
	if m.Iter == sol.IterMax {
		err = errors.New("Maximum number of Iterations reached")
	}

	return err
}
