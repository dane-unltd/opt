package uni

import (
	"errors"
	"math"
	"time"
)

//Exact line search for strictly quasi-convex functions
type Quadratic struct {
	TolRel  float64
	IterMax int
	TimeMax time.Duration
	Inexact bool
}

func NewQuadratic(inexact bool) *Quadratic {
	return &Quadratic{
		TolRel:  1e-2,
		IterMax: 1000,
		TimeMax: time.Second,
		Inexact: inexact,
	}
}

func (sol *Quadratic) Solve(m *Model) error {
	var err error = nil
	var eps float64

	tStart := time.Now()

	if math.IsNaN(m.ObjLB) {
		m.ObjLB = m.Obj(m.LB)
	}

	fNew := 0.0
	xNew := 0.0

	x0 := m.LB
	f0 := m.ObjLB
	g0 := m.DerivLB

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

		step := m.X - m.LB
		if fNew < m.ObjLB {
			m.ObjX = fNew
			m.X = xNew

			m.UB = m.X + step
			m.ObjUB = m.Obj(m.UB)
			for ; m.Iter < sol.IterMax && m.ObjUB <= m.ObjX; m.Iter++ {
				m.LB = m.X
				m.ObjLB = m.ObjX

				m.X = m.UB
				m.ObjX = m.ObjUB

				step *= 2
				m.UB = m.X + step
				m.ObjUB = m.Obj(m.UB)

				m.Time = time.Since(tStart)
				m.DoCallbacks()
				if m.Time > sol.TimeMax {
					err = errors.New("Quadratic: Time limit reached")
					goto done
				}
			}
		} else {
			m.ObjUB = fNew
			m.UB = xNew

			step *= 0.5

			m.X = m.LB + step
			m.ObjX = m.Obj(m.X)
			for ; m.Iter < sol.IterMax && m.ObjX >= m.ObjLB; m.Iter++ {
				m.UB = m.X
				m.ObjUB = m.ObjX

				step *= 0.5
				m.X = m.LB + step
				m.ObjX = m.Obj(m.X)

				m.Time = time.Since(tStart)
				m.DoCallbacks()
				if m.Time > sol.TimeMax {
					err = errors.New("Quadratic: Time limit reached")
					goto done
				}
			}
		}
	} else {
		eps = sol.TolRel * (m.UB - m.LB)
		m.UB = m.UB
		m.ObjUB = m.ObjUB
		if math.IsNaN(m.ObjUB) {
			m.ObjUB = m.Obj(m.UB)
		}
		if m.ObjUB < m.ObjLB {
			m.X = m.UB - eps
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
					err = errors.New("Quadratic: Time limit reached")
					goto done
				}
			}
		}
	}

	if eps == 0 {
		eps = sol.TolRel * (m.UB - m.LB)
	}

	for ; m.Iter < sol.IterMax; m.Iter++ {
		xNew = -0.5 * (m.UB*m.UB*(m.ObjLB-m.ObjX) + m.X*m.X*(m.ObjUB-m.ObjLB) + m.LB*m.LB*(m.ObjX-m.ObjUB)) /
			(m.UB*(m.ObjX-m.ObjLB) + m.X*(m.ObjLB-m.ObjUB) + m.LB*(m.ObjUB-m.ObjX))

		if math.Abs(m.LB-xNew) < 0.4*eps {
			xNew = m.LB + 0.4*eps
		}
		if math.Abs(m.UB-xNew) < 0.4*eps {
			xNew = m.UB - 0.4*eps
		}
		if math.Abs(m.X-xNew) < 0.4*eps {
			if m.UB-m.X > m.X-m.LB {
				xNew = m.X + 0.4*eps
			} else {
				xNew = m.X - 0.4*eps
			}
		}

		fNew = m.Obj(xNew)

		if !(xNew > m.LB && xNew < m.UB) || (xNew < m.X && fNew > m.ObjLB) ||
			(xNew > m.X && fNew > m.ObjUB) {
			//	err = errors.New("Quadratic: ran into numerical problems")
			break
		}

		if xNew > m.X {
			if fNew >= m.ObjX {
				m.UB = xNew
				m.ObjUB = fNew
			} else {
				m.LB = m.X
				m.ObjLB = m.ObjX
				m.X = xNew
				m.ObjX = fNew
			}
		} else {
			if fNew >= m.ObjX {
				m.LB = xNew
				m.ObjLB = fNew
			} else {
				m.UB = m.X
				m.ObjUB = m.ObjX
				m.X = xNew
				m.ObjX = fNew
			}
		}

		m.Time = time.Since(tStart)
		m.DoCallbacks()

		if sol.Inexact {
			if f0-m.ObjX <= (m.X-x0)*0.5*g0 {
				break
			}
		}
		if (m.UB - m.LB) <= eps {
			break
		}
		if m.Time > sol.TimeMax {
			err = errors.New("Quadratic: Time limit reached")
			break
		}
	}

done:
	if m.Iter == sol.IterMax {
		err = errors.New("Quadratic: Maximum number of Iterations reached")
	}

	return err
}
