

from scipy.integrate import solve_ivp 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.style.use('seaborn')

N =  100_000        # poblacion total
NT = N/5            # poblacion de trabajadores (1 por cada 5 habitantes)

#----------------------------
#   Variables generales
#----------------------------
CE = 1.5
tiempo_recuperacion = 3
periodo_incubacion = 14

#----------------------------
#   Variables trabajadores
#----------------------------
contacto_trabajadores = 2
mortalidad_trabajadores = 0.05

#-------------------------
#   Variables hospital
#-------------------------
camas_hospital = N/10             # 1 cama por cada 10 habitantes
camas_icu = N/20                  # 1 cama por cada 20 habitantes
pacientes_trabajador = 5          # cuántos pacientes puede atender un trabajador

tiempo_estadia = 10
tiempo_estadia_icu = 20

mortalidad_espera = 0.0
mortalidad_hospital = 0.0
mortalidad_icu = 0.0

recuperacion_espera = 0

fraccion_hospitalizacion = 0.05
fraccion_icu = 0.05

#-------------------------
#   Variables poblacion
#-------------------------
mortalidad = 0.05



def modelo_hospitales(t, y, d):
    '''
    Tengo 3 stocks principales: 
        - personas en espera (W)
        - personas en el hospital (H)
        - personas en ICU (ICU)
    Stocks adicionales:
        - recuperados (R)
        - muertos (M)
    Dependencias:
        - personas en espera depende del # infectados
        - admisiones x día depende de trabajadores de salud disponibles
    
    Parámetros:
        - t = tiempo 
        - y = stocks en tiempo t
        - d = dependendias en tiempo t (# infectados y # trabajadores disponibles)
    '''
    espera, hospitalizados, icu, recuperados, muertos = y
    infectados, trabajadores = d
    
    altas_dia_icu = icu / tiempo_estadia_icu
    altas_dia = hospitalizados / tiempo_estadia

    muertes_dia = (hospitalizados / tiempo_estadia_icu) * mortalidad_hospital
    muertes_dia_icu = icu * mortalidad_icu

    camas_disponibles = camas_hospital - hospitalizados + altas_dia + muertes_dia
    admisiones_dia = min(trabajadores*pacientes_trabajador, camas_disponibles)
    camas_disponibles_icu = camas_icu - icu + altas_dia_icu + muertes_dia_icu

    # Flujos
    I_W = infectados * fraccion_hospitalizacion
    W_M = espera * mortalidad_espera
    W_R = espera * recuperacion_espera
    W_H = min(espera, admisiones_dia)

    H_R = altas_dia
    H_ICU = min(hospitalizados*fraccion_icu, camas_disponibles_icu)
    H_M = muertes_dia

    ICU_R = altas_dia_icu
    ICU_M = muertes_dia_icu


    # Stocks
    dw = I_W - W_M - W_R - W_H
    dh = W_H - H_M - H_R - H_ICU
    dicu = H_ICU - ICU_M - ICU_R
    dr = W_R + H_R + ICU_R
    dm = W_M + H_M + ICU_M

    return (dw, dh, dicu, dr, dm)

def seirs_trabajadores(t, y):
    '''
    Tengo 5 stocks: 
        - Trabajadores susceptibles (S)
        - Trabajadores latentes (E)
        - Trabajadores infectados (I)
        - Trabajadores recuperados (R)
        - Trabajadores muertos (M)

    Parámetros:
        - t = tiempo 
        - y = stocks en tiempo t
    '''
    susceptibles, latentes, infectados, recuperados, muertos = y

    β = CE / N 
    λ = β * infectados * contacto_trabajadores

    # flujos
    S_E = susceptibles * λ
    E_I = latentes / periodo_incubacion
    I_M = infectados * mortalidad_trabajadores
    I_R = infectados / tiempo_recuperacion

    # stocks
    ds = - S_E
    de = S_E - E_I
    di = E_I - I_M - I_R
    dr = I_R
    dm = I_M

    return (ds, de, di, dr, dm)

def seirs_poblacion(t, y):
    '''
    Tengo 5 stocks: 
        - Susceptibles (S)
        - Latentes (E)
        - Infectados (I)
        - Recuperados (R)
        - Muertos (M)

    Parámetros:
        - t = tiempo 
        - y = stocks en tiempo t
    '''
    susceptibles, latentes, infectados, recuperados, muertos = y

    β = CE / N 
    λ = β * infectados

    # flujos
    S_E = susceptibles * λ
    E_I = latentes / periodo_incubacion
    I_M = infectados * mortalidad
    I_R = infectados / tiempo_recuperacion

    # stocks
    ds = - S_E
    de = S_E - E_I
    di = E_I - I_M - I_R
    dr = I_R
    dm = I_M

    return (ds, de, di, dr, dm)

def modelo_hospitalario(t, y):
    '''
    Asumimos que todos los infectados requieren hospitalización
    Stocks: 
        - Trabajadores susceptibles (TS)
        - Trabajadores latentes (TE)
        - Trabajadores infectados (TI)
        - Trabajadores recuperados (TR)
        - Trabajadores muertos (TM)
        - Susceptibles (S)
        - Latentes (E)
        - Infectados (I)
        - Recuperados (R)
        - Muertos (M)
        - Espera (W)
        - Hospitalizados (H)
        - ICU (ICU)

    Parámetros:
        - t = tiempo 
        - y = stocks en tiempo t
    '''

    t_susceptibles, t_latentes, t_infectados, t_recuperados, t_muertos, susceptibles, latentes, infectados, recuperados, muertos, espera, hospitalizados, icu  = y

    # Primero resuelvo para el modelo de trabajadores y el modelo epidemiológico porque no tienen dependencias
    
    # trabajadores
    y_t = (t_susceptibles, t_latentes, t_infectados, t_recuperados, t_muertos)
    dts, dte, dti, dtr, dtm = seirs_trabajadores(t, y_t)

    trabajadores_disponibles = dts + dte + dtr

    # población
    y_p = (susceptibles, latentes, infectados, recuperados, muertos)
    ds, de, di, dr, dm = seirs_poblacion(t, y_p)

    # modelo de hospitales
    y_h = (espera, hospitalizados, icu, recuperados, muertos)
    d = (di, trabajadores_disponibles)
    dw, dh, dicu, dr_h, dm_h = modelo_hospitales(t, y_h, d)
    
    # agrego al stock de recuperados y muertos los derivados del hospital
    dr += dr_h
    dm += dm_h

    return (dts, dte, dti, dtr, dtm, ds, de, di, dr, dm, dw, dh, dicu)


def main():
    # valores iniciales
    TS0 = NT
    TE0 = 0
    TI0 = 0
    TR0 = 0
    TM0 = 0
    
    I0 = 1 
    S0 = N - I0
    E0 = 0
    R0 = 0
    M0 = 0

    W0 = 0
    H0 = 0
    ICU0 = 0

    Y0 = (TS0, TE0, TI0, TR0, TM0, S0, E0, I0, R0, M0, W0, H0, ICU0)

    ts = 100
    sol = solve_ivp(modelo_hospitalario, [0,ts], Y0, t_eval=np.arange(0,ts,0.125), max_step=0.5)

if __name__ == '__main__':
    main()