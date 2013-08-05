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

	if math.IsNaN(m.objLB) {
		m.objLB = m.obj(m.lb)
	}

	fNew := 0.0
	xNew := 0.0

	m.iter = 0

	if math.IsInf(m.ub, 1) {
		xNew = m.x
		fNew = m.objX
		if math.IsNaN(xNew) {
			xNew = 1
			fNew = m.obj(xNew)
		}
		if math.IsNaN(fNew) {
			fNew = m.obj(xNew)
		}

		if fNew < m.objLB {
			m.objX = fNew
			m.x = xNew

			m.ub = 2 * m.x
			m.objUB = m.obj(m.ub)
			for ; m.iter < sol.IterMax && m.objUB <= m.objX; m.iter++ {
				m.ub *= 2
				m.objUB = m.obj(m.ub)

				m.time = time.Since(tStart)
				m.DoCallbacks()
				if m.time > sol.TimeMax {
					err = errors.New("Time limit reached")
					goto done
				}
			}
		} else {
			m.objUB = fNew
			m.ub = xNew

			m.x = 0.5 * m.ub
			m.objX = m.obj(m.x)
			for ; m.iter < sol.IterMax && m.objX >= m.objLB; m.iter++ {
				m.x *= 0.5
				m.objX = m.obj(m.x)

				m.time = time.Since(tStart)
				m.DoCallbacks()
				if m.time > sol.TimeMax {
					err = errors.New("Time limit reached")
					goto done
				}
			}
		}
	} else {
		m.ub = m.ub
		m.objUB = m.objUB
		if math.IsNaN(m.objUB) {
			m.objUB = m.obj(m.ub)
		}
		if m.objUB < m.objLB {
			m.x = m.ub - sol.Tol
			m.objX = m.obj(m.x)
			if m.objX >= m.objUB {
				m.x = m.ub
				m.objX = m.objUB
				goto done
			}
		} else {
			m.x = 0.5 * m.ub
			m.objX = m.obj(m.x)
			for ; m.iter < sol.IterMax && m.objX >= m.objLB; m.iter++ {
				m.x *= 0.5
				m.objX = m.obj(m.x)

				m.time = time.Since(tStart)
				m.DoCallbacks()
				if m.time > sol.TimeMax {
					err = errors.New("Time limit reached")
					goto done
				}
			}
		}
	}

	for ; m.ub-m.lb > sol.Tol && m.iter < sol.IterMax; m.iter++ {
		xNew = -0.5 * (m.ub*m.ub*(m.objLB-m.objX) + m.x*m.x*(m.objUB-m.objLB) + m.lb*m.lb*(m.objX-m.objUB)) /
			(m.ub*(m.objX-m.objLB) + m.x*(m.objLB-m.objUB) + m.lb*(m.objUB-m.objX))
		fNew = m.obj(xNew)
		if xNew > m.x {
			if fNew >= m.objX {
				m.ub = xNew
			} else {
				m.lb = m.x
				m.x = xNew
				m.objX = fNew
			}
		} else {
			if fNew >= m.objX {
				m.lb = xNew
			} else {
				m.ub = m.x
				m.x = xNew
				m.objX = fNew
			}
		}

		m.time = time.Since(tStart)
		m.DoCallbacks()
		if m.time > sol.TimeMax {
			err = errors.New("Time limit reached")
			goto done
		}
	}

done:
	if m.iter == sol.IterMax {
		err = errors.New("Maximum number of Iterations reached")
	}

	return err
}
