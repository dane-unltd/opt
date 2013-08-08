package uni

import (
	"math"
)

//Exact line search for strictly quasi-convex functions
type Quadratic struct {
	Inexact bool
}

func NewQuadratic(inexact bool) *Quadratic {
	return &Quadratic{
		Inexact: inexact,
	}
}

func (sol *Quadratic) Solve(m *Model) {
	var eps float64

	if math.IsNaN(m.ObjLB) {
		m.ObjLB = m.Obj.Val(m.LB)
	}

	fNew := 0.0
	xNew := 0.0

	x0 := m.LB
	f0 := m.ObjLB
	g0 := m.DerivLB

	m.init(false, false)

	if math.IsInf(m.UB, 1) {
		xNew = m.X
		fNew = m.ObjX
		if math.IsNaN(xNew) {
			xNew = 1
			fNew = m.Obj.Val(xNew)
		}
		if math.IsNaN(fNew) {
			fNew = m.Obj.Val(xNew)
		}

		step := m.X - m.LB
		if fNew < m.ObjLB {
			m.ObjX = fNew
			m.X = xNew

			m.UB = m.X + step
			m.ObjUB = m.Obj.Val(m.UB)
			for m.ObjUB <= m.ObjX {
				m.LB = m.X
				m.ObjLB = m.ObjX

				m.X = m.UB
				m.ObjX = m.ObjUB

				step *= 2
				m.UB = m.X + step
				m.ObjUB = m.Obj.Val(m.UB)

				if m.Status = m.update(); m.Status != 0 {
					return
				}
			}
		} else {
			m.ObjUB = fNew
			m.UB = xNew

			step *= 0.5

			m.X = m.LB + step
			m.ObjX = m.Obj.Val(m.X)
			for m.ObjX >= m.ObjLB {
				m.UB = m.X
				m.ObjUB = m.ObjX

				step *= 0.5
				m.X = m.LB + step
				m.ObjX = m.Obj.Val(m.X)

				if m.Status = m.update(); m.Status != 0 {
					return
				}
			}
		}
	} else {
		eps = math.Min(m.Params.XTolAbs, m.Params.XTolRel*m.initialInterval)
		if eps <= 0 {
			eps = 0.01 * m.initialInterval
		}
		m.UB = m.UB
		m.ObjUB = m.ObjUB
		if math.IsNaN(m.ObjUB) {
			m.ObjUB = m.Obj.Val(m.UB)
		}
		if m.ObjUB < m.ObjLB {
			m.X = m.UB - eps
			m.ObjX = m.Obj.Val(m.X)
			if m.ObjX >= m.ObjUB {
				m.X = m.UB
				m.ObjX = m.ObjUB
				m.Status = XRelConv
				return
			}
		} else {
			m.X = 0.5 * m.UB
			m.ObjX = m.Obj.Val(m.X)
			for m.ObjX >= m.ObjLB {
				m.X *= 0.5
				m.ObjX = m.Obj.Val(m.X)

				if m.Status = m.update(); m.Status != 0 {
					return
				}
			}
		}
	}

	if eps == 0 {
		m.initialInterval = m.UB - m.LB
		eps = math.Min(m.Params.XTolAbs, m.Params.XTolRel*m.initialInterval)
		if eps == 0 {
			eps = 0.01 * m.initialInterval
		}
	}

	for {
		//optimum of quadratic fit
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

		fNew = m.Obj.Val(xNew)

		if !(xNew > m.LB && xNew < m.UB) || (xNew < m.X && fNew > m.ObjLB) ||
			(xNew > m.X && fNew > m.ObjUB) {
			if m.UB-m.X > m.X-m.LB {
				xNew = (m.X + m.UB) / 2
			} else {
				xNew = (m.X + m.LB) / 2
			}
			fNew = m.Obj.Val(xNew)
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

		if sol.Inexact {
			if f0-m.ObjX <= (m.X-x0)*0.5*g0 {
				break
			}
		}
		if m.Status = m.update(); m.Status != 0 {
			break
		}
	}
}
