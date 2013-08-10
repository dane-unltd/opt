package uni

import (
	"math"
)

type Cubic struct {
}

func NewCubic() *Cubic {
	return &Cubic{}
}

func (sol *Cubic) Solve(m *Model) {
	m.init(true, false)

	eps := 0.4 * m.Params.XTolAbs

	if m.DerivLB > 0 {
		m.Status = Fail
		return
	}

	//search for upper bound
	for math.IsInf(m.UB, 1) {
		if m.DerivX > 0 {
			m.UB = m.X
			m.ObjUB = m.ObjX
			m.DerivUB = m.DerivX
		} else {
			if m.ObjX-m.f0 > m.Params.Armijo*m.d0*(m.X-m.x0) {
				m.UB = m.X
				m.ObjUB = m.ObjX
				m.DerivUB = m.DerivX
			} else {
				lb := m.LB
				m.LB = m.X
				m.ObjLB = m.ObjX
				m.DerivLB = m.DerivX

				m.X += 2 * (m.X - lb)
				m.ObjX, m.DerivX = m.deriv.ValDeriv(m.X)
				m.FunEvals++
				m.DerivEvals++
				if m.Status = m.update(); m.Status != 0 {
					return
				}
			}
		}
	}

	if eps == 0 {
		eps = m.Params.XTolRel * (m.UB - m.LB)
	}
	if eps == 0 {
		eps = 1e-3 * (m.UB - m.LB)
	}

	for {
		//cubic interpolation between upper bound and lower bound
		eta := 3*(m.ObjLB-m.ObjUB)/(m.UB-m.LB) + m.DerivLB + m.DerivUB
		nu := math.Sqrt(math.Pow(eta, 2) - m.DerivLB*m.DerivUB)

		m.X = m.LB + (m.UB-m.LB)*
			(1-(m.DerivUB+nu-eta)/(m.DerivUB-m.DerivLB+2*nu))

		if !(m.X > m.LB && m.X < m.UB) {
			m.X = (m.UB + m.LB) / 2
		}
		if (m.X - m.LB) < eps {
			m.X = m.LB + eps
		}
		if (m.UB - m.X) < eps {
			m.X = m.UB - eps
		}

		m.ObjX, m.DerivX = m.deriv.ValDeriv(m.X)
		m.FunEvals++
		m.DerivEvals++

		if m.DerivX > 0 {
			m.UB = m.X
			m.ObjUB = m.ObjX
			m.DerivUB = m.DerivX
		} else {
			if m.ObjX-m.f0 > m.Params.Armijo*m.d0*(m.X-m.x0) {
				m.UB = m.X
				m.ObjUB = m.ObjX
				m.DerivUB = m.DerivX
			} else {
				m.LB = m.X
				m.ObjLB = m.ObjX
				m.DerivLB = m.DerivX
			}
		}

		if m.Status = m.update(); m.Status != 0 {
			return
		}
	}
}
