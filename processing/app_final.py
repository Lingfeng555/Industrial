import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from pathlib import Path
from PIL import Image
import requests
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, f1_score, recall_score
import bentoml

# CÓDIGO DE ENTRENAMIENTO DE MODELOS
def train_models_tab():
    st.header("🤖 Entrenamiento de Modelos de Regresión")
    
    # 1. Cargar datos
    @st.cache_data
    def load_data():
        return pd.read_csv("./data_augmented.csv")
    
    df = load_data()
    st.success(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # 2. Selección de características y target
    st.subheader("🎯 Configurar Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Seleccionar características (X):**")
        exclude = ['sequence_id', 'frame_id', 'timestamp', 'time_seconds']
        available_features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.int64, np.float64]]
        
        selected_features = st.multiselect(
            "Características para el modelo:",
            available_features,
            default=['lidar_n_points_under_100m', 'count_Car', 'pose_y'] if 'lidar_n_points_under_100m' in available_features else available_features[:3]
        )
    
    with col2:
        st.write("**Seleccionar target (y):**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target = st.selectbox("Variable a predecir:", numeric_cols, index=numeric_cols.index('speed_y') if 'speed_y' in numeric_cols else 0)
    
    # 3. Configurar modelo
    st.subheader("⚙️ Configurar Entrenamiento")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Tamaño test set (%):", 10, 40, 20) / 100
        random_state = st.number_input("Random state:", 0, 100, 42)
    
    with col2:
        model_type = st.selectbox("Tipo de modelo:", ["Random Forest", "Linear Regression"])
        
        if model_type == "Random Forest":
            n_estimators = st.slider("Número de árboles:", 10, 200, 100)
            max_depth = st.slider("Profundidad máxima:", 3, 20, 10)
    
    with col3:
        scale_features = st.checkbox("Estandarizar características", value=True)
        show_importance = st.checkbox("Mostrar importancia de características", value=True)
    
    # Función para convertir regresión a clasificación binaria para calcular F1 y Recall
    def regression_to_binary_classification(y_true, y_pred, threshold_percent=0.1):
        """
        Convierte predicciones de regresión a clasificación binaria.
        threshold_percent: porcentaje del rango para considerar como predicción correcta
        """
        y_range = y_true.max() - y_true.min()
        threshold = threshold_percent * y_range
        
        # Crear etiquetas binarias: 1 si la predicción está dentro del threshold, 0 si no
        y_true_binary = np.ones(len(y_true))  # Todas las predicciones son positivas en este contexto
        y_pred_binary = (np.abs(y_true - y_pred) <= threshold).astype(int)
        
        return y_true_binary, y_pred_binary
    
    # 4. Botón para entrenar
    if st.button("🚀 Entrenar Modelo", type="primary", width='stretch'):
        if not selected_features:
            st.error("❌ Selecciona al menos una característica")
        else:
            with st.spinner("Entrenando modelo..."):
                # Preparar datos
                X = df[selected_features].fillna(0)
                y = df[target]
                
                # Dividir train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Estandarizar si se solicita
                if scale_features:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                
                # Crear y entrenar modelo
                if model_type == "Random Forest":
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_jobs=-1
                    )
                else:  # Linear Regression
                    model = LinearRegression()
                
                model.fit(X_train, y_train)
                
                # Predecir
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calcular métricas de regresión
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                # Calcular métricas de clasificación (F1, Recall)
                # Convertir a problema binario para poder calcular estas métricas
                y_train_true_binary, y_train_pred_binary = regression_to_binary_classification(y_train, y_train_pred)
                y_test_true_binary, y_test_pred_binary = regression_to_binary_classification(y_test, y_test_pred)
                
                # Calcular F1 Score
                train_f1 = f1_score(y_train_true_binary, y_train_pred_binary, zero_division=0)
                test_f1 = f1_score(y_test_true_binary, y_test_pred_binary, zero_division=0)
                
                # Calcular Recall
                train_recall = recall_score(y_train_true_binary, y_train_pred_binary, zero_division=0)
                test_recall = recall_score(y_test_true_binary, y_test_pred_binary, zero_division=0)
                
                # Mostrar resultados
                st.subheader("📊 Resultados del Entrenamiento")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MAE Train", f"{train_mae:.4f}")
                    st.metric("R² Train", f"{train_r2:.4f}")
                
                with col2:
                    st.metric("MAE Test", f"{test_mae:.4f}")
                    st.metric("R² Test", f"{test_r2:.4f}")
                
                with col3:
                    st.metric("F1 Train", f"{train_f1:.4f}")
                    st.metric("Recall Train", f"{train_recall:.4f}")
                
                with col4:
                    st.metric("F1 Test", f"{test_f1:.4f}")
                    st.metric("Recall Test", f"{test_recall:.4f}")
                
                # Nota sobre cómo se calculan las métricas
                st.info("""
                **Nota:** Las métricas F1 y Recall se calculan convirtiendo el problema de regresión a clasificación binaria.
                Una predicción se considera "correcta" si está dentro del 10% del rango total del target.
                """)
                
                # Gráfico de predicciones vs reales
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Train
                ax1.scatter(y_train, y_train_pred, alpha=0.5, s=10)
                ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
                ax1.set_xlabel(f'Real ({target})')
                ax1.set_ylabel(f'Predicho ({target})')
                ax1.set_title(f'Train - R²={train_r2:.3f}')
                ax1.grid(True, alpha=0.3)
                
                # Test
                ax2.scatter(y_test, y_test_pred, alpha=0.5, s=10)
                ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax2.set_xlabel(f'Real ({target})')
                ax2.set_ylabel(f'Predicho ({target})')
                ax2.set_title(f'Test - R²={test_r2:.3f}')
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Importancia de características (solo para Random Forest)
                if show_importance and hasattr(model, 'feature_importances_'):
                    st.subheader("📈 Importancia de Características")
                    
                    importance_df = pd.DataFrame({
                        'Característica': selected_features,
                        'Importancia': model.feature_importances_
                    }).sort_values('Importancia', ascending=False)
                    
                    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                    ax_imp.barh(importance_df['Característica'], importance_df['Importancia'])
                    ax_imp.set_xlabel('Importancia')
                    ax_imp.set_title('Importancia de Características - Random Forest')
                    plt.tight_layout()
                    st.pyplot(fig_imp)
                    
                    st.dataframe(importance_df, width='stretch')
                
                # Información del modelo
                with st.expander("📋 Información del Modelo", expanded=False):
                    st.write(f"**Modelo:** {model_type}")
                    st.write(f"**Características usadas:** {len(selected_features)}")
                    st.write(f"**Tamaño train:** {len(X_train)} muestras")
                    st.write(f"**Tamaño test:** {len(X_test)} muestras")
                    st.write(f"**Target:** {target}")
                    
                    if hasattr(model, 'coef_'):
                        coef_df = pd.DataFrame({
                            'Característica': selected_features,
                            'Coeficiente': model.coef_
                        })
                        st.write("**Coeficientes (Linear Regression):**")
                        st.dataframe(coef_df, width='stretch')
                    
                    # Mostrar estadísticas de las métricas de clasificación
                    st.write("**Estadísticas de clasificación binaria:**")
                    st.write(f"- Threshold usado: 10% del rango total ({target})")
                    st.write(f"- Predicciones correctas (Train): {sum(y_train_pred_binary)}/{len(y_train_pred_binary)}")
                    st.write(f"- Predicciones correctas (Test): {sum(y_test_pred_binary)}/{len(y_test_pred_binary)}")
    
    # 5. Información de ayuda
    with st.expander("ℹ️ Cómo usar esta app", expanded=True):
        st.write("""
        1. **Selecciona características** - Variables que el modelo usará para predecir
        2. **Elige target** - Variable que quieres predecir (ej: speed_y)
        3. **Configura entrenamiento** - Tamaño test, tipo de modelo, etc.
        4. **Haz clic en 'Entrenar Modelo'** - ¡Listo!
        
        **Métricas explicadas:**
        - **MAE**: Error Absoluto Medio - promedio de errores absolutos
        - **R²**: Coeficiente de determinación - proporción de varianza explicada
        - **F1**: Media armónica de precisión y recall
        - **Recall**: Proporción de predicciones correctamente identificadas
        
        **Nota sobre F1 y Recall:**
        Estas métricas son típicas de clasificación. Para regresión, convertimos el problema 
        a clasificación binaria considerando una predicción como "correcta" si está dentro 
        del 10% del rango total de la variable target.
        
        **Consejos:**
        - Empieza con pocas características
        - Prueba diferentes targets
        - Compara Random Forest vs Linear Regression
        - Si R² test es muy bajo, revisa las características
        """)
    
    # 6. Vista previa de datos
    with st.expander("👁️ Vista previa de datos", expanded=False):
        st.write(f"**Primeras 5 filas de datos seleccionados:**")
        if selected_features:
            preview_cols = selected_features + [target] if target not in selected_features else selected_features
            st.dataframe(df[preview_cols].head(), width='stretch')
        else:
            st.dataframe(df.head(), width='stretch')
        
        st.write(f"**Estadísticas de target ({target}):**")
        st.write(f"- Promedio: {df[target].mean():.4f}")
        st.write(f"- Desviación: {df[target].std():.4f}")
        st.write(f"- Mínimo: {df[target].min():.4f}")
        st.write(f"- Máximo: {df[target].max():.4f}")
        st.write(f"- Rango: {df[target].max() - df[target].min():.4f}")

#CÓDIGO DE BENTOML
def bento_tab():
    st.header("🚗 Predicción de Velocidad con BentoML")
    
    speed_prediction_client = bentoml.SyncHTTPClient('http://localhost:3002')
    # --- Configuración ---
    BENTO_API_URL = "http://localhost:3002/predict_speed"  # Updated port for SpeedPredictionService
    TEST_FEATURES_PATH = "./test_bent.csv"  # Features para predicción
    TEST_TARGET_PATH = "./test_bent_y.csv"  # Valores reales de speed_y
    
    # Las 7 features que tu modelo necesita
    TOP_7_FEATURES = ['speed_y_avg_20', 'speed_x', 'speed_x_avg_20', 
                      'lidar_n_points_total', 'lidar_intensity_mean', 
                      'lidar_n_points_under_100m', 'lidar_points_10_20m']
    
    @st.cache_data
    def load_test_data(features_path, target_path):
        """Carga ambos archivos de test y los combina"""
        try:
            # Cargar features
            features_df = pd.read_csv(features_path)
            
            # Cargar target (speed_y real)
            target_df = pd.read_csv(target_path)
            
            # Verificar que tengan el mismo número de filas
            if len(features_df) != len(target_df):
                st.warning(f"Número de filas diferente: Features={len(features_df)}, Target={len(target_df)}")
                min_len = min(len(features_df), len(target_df))
                features_df = features_df.iloc[:min_len]
                target_df = target_df.iloc[:min_len]
            
            # Combinar
            features_df['speed_y_real'] = target_df['speed_y'] if 'speed_y' in target_df.columns else target_df.iloc[:, 0]
            
            return features_df
        
        except FileNotFoundError as e:
            st.error(f"Error: No se encuentra el archivo '{e.filename}'")
            return None
        except Exception as e:
            st.error(f"Error al cargar los datos: {e}")
            return None
    
    # Cargar datos
    test_df = load_test_data(TEST_FEATURES_PATH, TEST_TARGET_PATH)
    
    if test_df is not None:
        # Sidebar para configuración
        with st.sidebar:
            st.header("⚙️ Configuración")
            
            # Métricas globales
            st.subheader("📊 Estadísticas del Dataset")
            st.metric("Filas totales", len(test_df))
            st.metric("Velocidad promedio real", f"{test_df['speed_y_real'].mean():.2f} m/s")
            st.metric("Velocidad máxima real", f"{test_df['speed_y_real'].max():.2f} m/s")
            
            # Historial de predicciones
            st.subheader("📈 Historial")
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            
            if st.button("🗑️ Limpiar historial"):
                st.session_state.prediction_history = []
                st.rerun()
        
        # Layout principal
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🔍 Selección de Datos")
            
            # 1. Selector de fila con slider
            row_index = st.slider(
                f"Selecciona un índice de fila:",
                min_value=0,
                max_value=len(test_df) - 1,
                value=min(100, len(test_df) - 1),
                help="Desliza para seleccionar diferentes datos de test"
            )
            
            selected_row = test_df.iloc[[row_index]]
            
            # Mostrar datos seleccionados en tabs
            tab1, tab2, tab3 = st.tabs(["📋 Datos Completos", "🎯 Features Clave", "📊 Valores"])
            
            with tab1:
                st.dataframe(selected_row, width='stretch')
            
            with tab2:
                # Solo las 7 features del modelo
                features_data = selected_row[TOP_7_FEATURES]
                st.dataframe(features_data, width='stretch')
                
                # Gráfico de barras de las features
                fig = go.Figure(data=[
                    go.Bar(x=TOP_7_FEATURES, y=features_data.values[0], 
                          marker_color='lightblue', name='Valores')
                ])
                fig.update_layout(
                    title="Valores de las 7 Features Clave",
                    xaxis_title="Feature",
                    yaxis_title="Valor",
                    height=300
                )
                st.plotly_chart(fig, width='stretch')
            
            with tab3:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Velocidad real actual", 
                             f"{selected_row['speed_y_real'].values[0]:.3f} m/s",
                             f"{selected_row['speed_y_real'].values[0]*3.6:.1f} km/h")
                
                # Mostrar estadísticas de la fila
                st.write("**Resumen de features:**")
                stats_df = pd.DataFrame({
                    'Estadística': ['Mínimo', 'Promedio', 'Máximo'],
                    'Valor': [
                        features_data.min().min(),
                        features_data.mean().mean(),
                        features_data.max().max()
                    ]
                })
                st.dataframe(stats_df, width='stretch', hide_index=True)
        
        with col2:
            st.header("🎯 Predicción")
            
            # Botón grande para predicción
            if st.button("**PREDECIR VELOCIDAD**", type="primary", width='stretch'):
                with st.spinner("Consultando modelo BentoML..."):
                    try:
                        # Preparar datos para la API
                        data_for_prediction = selected_row[TOP_7_FEATURES]
                        input_json = data_for_prediction.to_numpy()
                        
                        # Llamar a la API   
                        print("Enviando datos a la API de BentoML...")
                        response = speed_prediction_client.predict_speed(input_data=input_json).tolist()[0]
                        print(response)
                        if True:
                            # Obtener predicción
                            speed_prediction = response
                            real_speed = selected_row['speed_y_real'].values[0]
                            
                            # Guardar en historial
                            st.session_state.prediction_history.append({
                                'index': row_index,
                                'predicted': speed_prediction,
                                'real': real_speed,
                                'error': abs(speed_prediction - real_speed)
                            })
                            
                            # Mostrar resultados
                            st.success("✅ Predicción exitosa!")
                            
                            # Métricas en columnas
                            col_pred, col_real, col_error = st.columns(3)
                            
                            with col_pred:
                                st.metric(
                                    "Predicción", 
                                    f"{speed_prediction:.3f} m/s",
                                    f"{speed_prediction*3.6:.1f} km/h",
                                    delta_color="off"
                                )
                            
                            with col_real:
                                delta = real_speed - speed_prediction
                                st.metric(
                                    "Valor Real",
                                    f"{real_speed:.3f} m/s",
                                    f"{real_speed*3.6:.1f} km/h",
                                    delta=f"{delta:.3f} m/s",
                                    delta_color="inverse"
                                )
                            
                            with col_error:
                                error_percent = (abs(delta) / real_speed * 100) if real_speed != 0 else 0
                                st.metric(
                                    "Error Absoluto",
                                    f"{abs(delta):.3f} m/s",
                                    f"{error_percent:.1f}%",
                                    delta_color="off"
                                )
                            
                            # Gráfico de comparación
                            fig = go.Figure()
                            fig.add_trace(go.Indicator(
                                mode="gauge+number+delta",
                                value=speed_prediction,
                                delta={'reference': real_speed, 'relative': False},
                                gauge={
                                    'axis': {'range': [None, max(speed_prediction, real_speed)*1.5]},
                                    'bar': {'color': "lightblue"},
                                    'steps': [
                                        {'range': [0, real_speed*0.5], 'color': "lightgray"},
                                        {'range': [real_speed*0.5, real_speed], 'color': "gray"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': real_speed
                                    }
                                }
                            ))
                            fig.update_layout(
                                title="Comparación: Predicción vs Real",
                                height=300
                            )
                            st.plotly_chart(fig, width='stretch')
                            
                        else:
                            st.error(f"Error de la API ({response.status_code}): {response.text}")
                    
                    except requests.exceptions.ConnectionError:
                        st.error("🔌 Error de Conexión")
                        st.info("""
                        Asegúrate de que el servidor BentoML está corriendo:
                        ```bash
                        bentoml serve once_service:svc --port 3000
                        ```
                        """)
                    except requests.exceptions.Timeout:
                        st.error("Timeout - La API tardó demasiado en responder")
                    except Exception as e:
                        print(e)
                        pass
            else:
                st.info("👆 Haz clic en el botón para predecir la velocidad")
            
            # Historial de predicciones
            if st.session_state.prediction_history:
                st.subheader("📜 Historial de Predicciones")
                
                hist_df = pd.DataFrame(st.session_state.prediction_history)
                
                # Gráfico de líneas
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=hist_df.index, y=hist_df['predicted'],
                    mode='lines+markers', name='Predicción',
                    line=dict(color='blue')
                ))
                fig_line.add_trace(go.Scatter(
                    x=hist_df.index, y=hist_df['real'],
                    mode='lines+markers', name='Real',
                    line=dict(color='green', dash='dash')
                ))
                fig_line.update_layout(
                    title="Evolución de Predicciones",
                    xaxis_title="Número de predicción",
                    yaxis_title="Velocidad (m/s)",
                    height=250
                )
                st.plotly_chart(fig_line, width='stretch')
                
                print(hist_df.tail(5))
                # Tabla resumen
                st.dataframe(
                    hist_df.tail(5).style.format({
                        'predicted': '{:.3f}',
                        'real': '{:.3f}',
                        'error': '{:.3f}'
                    }),
                    width='stretch',
                    hide_index=True
                )
        
        # Sección inferior: Análisis de error
        st.markdown("---")
        st.header("📊 Análisis de Error")
        
        if st.session_state.prediction_history and len(st.session_state.prediction_history) > 1:
            hist_df = pd.DataFrame(st.session_state.prediction_history)
            
            col_err1, col_err2, col_err3 = st.columns(3)
            
            with col_err1:
                avg_error = hist_df['error'].mean()
                st.metric("Error promedio", f"{avg_error:.3f} m/s")
            
            with col_err2:
                max_error = hist_df['error'].max()
                st.metric("Error máximo", f"{max_error:.3f} m/s")
            
            with col_err3:
                mae = hist_df['error'].mean()
                st.metric("MAE", f"{mae:.3f} m/s")
            
            # Distribución de errores
            fig_hist = go.Figure(data=[go.Histogram(x=hist_df['error'], nbinsx=20)])
            fig_hist.update_layout(
                title="Distribución de Errores Absolutos",
                xaxis_title="Error (m/s)",
                yaxis_title="Frecuencia",
                height=300
            )
            st.plotly_chart(fig_hist, width='stretch')
        
        else:
            st.info("Realiza varias predicciones para ver el análisis de error")
    
    # Mensaje si no hay datos
    else:
        st.error("""
        No se pudieron cargar los datos de test.
        
        Asegúrate de que existen los archivos:
        1. `./test_bent.csv` - Features para predicción
        2. `./test_bent_y.csv` - Valores reales de speed_y
        
        El archivo de features debe contener al menos estas columnas:
        ```python
        ['speed_y_avg_20', 'speed_x', 'speed_x_avg_20', 
         'lidar_n_points_total', 'lidar_intensity_mean', 
         'lidar_n_points_under_100m', 'lidar_points_10_20m']
        ```
        """)

# PÁGINA PRINCIPAL 
# Cargar datos originales 
@st.cache_data
def load_data():
    return pd.read_csv("./data.csv")

df_total = load_data()

st.set_page_config(
    page_title="ONCE Dashboard",
    page_icon="🚗",
    layout="wide"
)

# # Título principal 
# st.title("🚗 Dashboard ONCE - Análisis por Secuencia")
# st.markdown("---")

