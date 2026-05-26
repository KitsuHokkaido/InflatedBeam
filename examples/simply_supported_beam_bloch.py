import numpy as np

def run_calculation(R_nat, L_nat, v_nat, E_nat, p, H_0, F):
    G_nat = E_nat/(2*(1 + v_nat))

    R_0 = R_nat + ((p*R_nat**2)*(2 - v_nat))/(2*E_nat*H_0)
    print(f"R_0 : {R_0}")

    L_0 = L_nat + ((p*R_nat*L_nat)*(2 - v_nat))/(2*E_nat*H_0)
    print(f"L_0 : {L_0}")
    k_x = L_0/L_nat 

    k_theta = R_0/R_nat

    E_0 = (E_nat*k_x**3)/k_theta
    print(f"E_0 : {E_0}")

    G_0 = k_x*k_theta*G_nat
    v = E_0/(2*G_0) - 1
    print(f"v : {v}")
    
    I_0 = np.pi * R_0**3 * H_0 

    S_0 = 2*np.pi*R_0*H_0 

    P = p*np.pi*R_0**2

    k = 1/2

    def V(x):
        return ((F*x)*(3*L_0**2 - 4*x**2))/(48*(E_0*I_0 + (P*I_0)/S_0)) + (F*x)/(2*(P + k*G_0*S_0))

    def theta_left(x):
        return (F*(L_0**2 - 4*x**2))/(16*(E_0*I_0 +(P*I_0)/S_0))

    def theta_right(x):
        return (F*(4*(L_0-x)**2 - L_0**2))/(16*(E_0*I_0 + (P*I_0)/S_0))
    
    print(f"Déplacement en x = {L_0/2}, v = {V(L_0/2)}")
    print(f"Rotation en x = {0}, theta = {theta_left(0)}")
    print(f"Rotation en x = {L_0}, theta = {theta_right(L_0)}")
    

if __name__ == '__main__':
    run_calculation(3.0, 60, 0.3, 400, 1, 0.05, 1)    
