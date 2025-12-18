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
    if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
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
                    
                    st.dataframe(importance_df, use_container_width=True)
                
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
                        st.dataframe(coef_df, use_container_width=True)
                    
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
            st.dataframe(df[preview_cols].head(), use_container_width=True)
        else:
            st.dataframe(df.head(), use_container_width=True)
        
        st.write(f"**Estadísticas de target ({target}):**")
        st.write(f"- Promedio: {df[target].mean():.4f}")
        st.write(f"- Desviación: {df[target].std():.4f}")
        st.write(f"- Mínimo: {df[target].min():.4f}")
        st.write(f"- Máximo: {df[target].max():.4f}")
        st.write(f"- Rango: {df[target].max() - df[target].min():.4f}")

#CÓDIGO DE BENTOML
def bento_tab():
    st.header("🚗 Predicción de Velocidad con BentoML")
    
    # --- Configuración ---
    BENTO_API_URL = "http://localhost:3000/predict_speed"  # Cambia a tu endpoint
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
                st.dataframe(selected_row, use_container_width=True)
            
            with tab2:
                # Solo las 7 features del modelo
                features_data = selected_row[TOP_7_FEATURES]
                st.dataframe(features_data, use_container_width=True)
                
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
                st.plotly_chart(fig, use_container_width=True)
            
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
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.header("🎯 Predicción")
            
            # Botón grande para predicción
            if st.button("**PREDECIR VELOCIDAD**", type="primary", use_container_width=True):
                with st.spinner("Consultando modelo BentoML..."):
                    try:
                        # Preparar datos para la API
                        data_for_prediction = selected_row[TOP_7_FEATURES]
                        input_json = data_for_prediction.to_numpy().tolist()
                        
                        # Llamar a la API
                        response = requests.post(
                            BENTO_API_URL, 
                            json=input_json,
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            # Obtener predicción
                            speed_prediction = response.json()[0]  # Ajusta según tu API
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
                            st.plotly_chart(fig, use_container_width=True)
                            
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
                st.plotly_chart(fig_line, use_container_width=True)
                
                # Tabla resumen
                st.dataframe(
                    hist_df.tail(5).style.format({
                        'predicted': '{:.3f}',
                        'real': '{:.3f}',
                        'error': '{:.3f}'
                    }),
                    use_container_width=True,
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
            st.plotly_chart(fig_hist, use_container_width=True)
        
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

# Título principal 
st.title("🚗 Dashboard ONCE - Análisis por Secuencia")
st.markdown("---")

# TABS (7)
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Matriz Correlación", 
    "📍 Trayectoria y Velocidad", 
    "📈 Análisis Detallado",
    "📋 Comparativas Globales",
    "🔍 Limpieza LiDAR",
    "🤖 Entrenar Modelos",  
    "🚗 BentoML Predictor"  
])



# Tab 1: Matriz de Correlación 
with tab1:
    st.header("📊 Matriz de Correlación por Secuencia")
    
    # Selector de secuencia
    available_seqs = sorted(df_total['sequence_id'].unique())
    selected_seq = st.selectbox("Selecciona una secuencia:", available_seqs, key="corr_seq")
    
    if selected_seq:
        df_seq = df_total[df_total['sequence_id'] == selected_seq].copy()
        
        # Seleccionar solo columnas numéricas
        numeric_cols = df_seq.select_dtypes(include=[np.number]).columns
        
        # Calcular matriz de correlación
        corr_matrix = df_seq[numeric_cols].corr()
        
        # Visualizar con seaborn
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(f'Matriz de Correlación - Secuencia {selected_seq}', fontsize=16, pad=20)
        
        st.pyplot(fig)
        
        # Mostrar correlaciones fuertes
        st.subheader("🔗 Correlaciones Fuertes (>0.7 o <-0.7)")
        strong_corr = corr_matrix[(corr_matrix.abs() > 0.7) & (corr_matrix != 1.0)].stack()
        
        if not strong_corr.empty:
            strong_corr_df = strong_corr.reset_index()
            strong_corr_df.columns = ['Variable 1', 'Variable 2', 'Correlación']
            strong_corr_df = strong_corr_df.drop_duplicates(subset=['Correlación'])
            strong_corr_df = strong_corr_df.sort_values('Correlación', ascending=False)
            st.dataframe(strong_corr_df, use_container_width=True)
        else:
            st.info("No hay correlaciones fuertes (>0.7) en esta secuencia") # solo estas

# Tab 2: Trayectoria y Velocidad 
with tab2:
    st.header("📍 Trayectoria y Velocidad por Secuencia")
    
    # Selector de secuencia
    selected_seq2 = st.selectbox("Selecciona una secuencia:", available_seqs, key="traj_seq")
    
    if selected_seq2:
        df_seq = df_total[df_total['sequence_id'] == selected_seq2].copy()
        df_seq = df_seq.sort_values('frame_id')
        
        # Convertir timestamps
        df_seq['timestamp'] = pd.to_datetime(df_seq['frame_id'], unit='ms')
        df_seq['time_seconds'] = (df_seq['timestamp'] - df_seq['timestamp'].iloc[0]).dt.total_seconds()
        
        # Calcular velocidad
        df_seq['delta_time'] = df_seq['time_seconds'].diff()
        df_seq['delta_y'] = df_seq['pose_y'].diff()
        df_seq['speed_y'] = df_seq['delta_y'] / df_seq['delta_time']
        
        # Mostrar estadísticas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Frames", len(df_seq))
        with col2:
            st.metric("Duración", f"{df_seq['time_seconds'].iloc[-1]:.0f}s")
        with col3:
            st.metric("Velocidad prom", f"{df_seq['speed_y'].abs().mean()*3.6:.1f} km/h")
        with col4:
            dist = np.sqrt((df_seq['pose_x'].iloc[-1] - df_seq['pose_x'].iloc[0])**2 + 
                          (df_seq['pose_y'].iloc[-1] - df_seq['pose_y'].iloc[0])**2)
            st.metric("Distancia", f"{dist:.0f}m")
        
        # Gráficos
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Trayectoria XY
        ax1 = plt.subplot(231)
        ax1.plot(df_seq['pose_x'], df_seq['pose_y'], 'b-', alpha=0.7, linewidth=1)
        ax1.scatter(df_seq['pose_x'].iloc[0], df_seq['pose_y'].iloc[0], 
                   color='green', s=100, label='Inicio')
        ax1.scatter(df_seq['pose_x'].iloc[-1], df_seq['pose_y'].iloc[-1], 
                   color='red', s=100, label='Fin')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Trayectoria del vehículo')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. Posición Y vs Tiempo
        ax2 = plt.subplot(232)
        ax2.plot(df_seq['time_seconds'], df_seq['pose_y'], 'g-', linewidth=2)
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Avance en Y vs Tiempo')
        ax2.grid(True, alpha=0.3)
        
        # 3. Velocidad en Y
        ax3 = plt.subplot(233)
        ax3.plot(df_seq['time_seconds'], df_seq['speed_y'], 'r-', alpha=0.7)
        ax3.axhline(y=df_seq['speed_y'].mean(), color='k', linestyle='--', 
                   label=f'Promedio: {df_seq["speed_y"].mean():.1f} m/s')
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel('Velocidad (m/s)')
        ax3.set_title('Velocidad del vehículo')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Trayectoria 3D
        ax4 = plt.subplot(234, projection='3d')
        ax4.plot(df_seq['pose_x'], df_seq['pose_y'], df_seq['pose_z'], 'b-', alpha=0.7)
        ax4.scatter(df_seq['pose_x'].iloc[0], df_seq['pose_y'].iloc[0], df_seq['pose_z'].iloc[0], 
                   color='green', s=100, label='Inicio')
        ax4.scatter(df_seq['pose_x'].iloc[-1], df_seq['pose_y'].iloc[-1], df_seq['pose_z'].iloc[-1], 
                   color='red', s=100, label='Fin')
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_zlabel('Z (m)')
        ax4.set_title('Trayectoria 3D')
        ax4.legend()
        
        # 5. Mapa de calor velocidad-posición
        ax5 = plt.subplot(235)
        scatter = ax5.scatter(df_seq['pose_x'], df_seq['pose_y'], 
                            c=df_seq['speed_y'].abs() * 3.6, 
                            cmap='RdYlGn', s=20, alpha=0.7, vmin=0, vmax=30)
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Y (m)')
        ax5.set_title('Velocidad en la trayectoria (km/h)')
        ax5.set_aspect('equal')
        plt.colorbar(scatter, ax=ax5, label='Velocidad (km/h)')
        
        # 6. Distribución de velocidades
        ax6 = plt.subplot(236)
        ax6.hist(df_seq['speed_y'].abs() * 3.6, bins=30, edgecolor='black', alpha=0.7)
        ax6.axvline(x=df_seq['speed_y'].abs().mean() * 3.6, color='red', linestyle='--',
                   label=f'Promedio: {df_seq["speed_y"].abs().mean()*3.6:.1f} km/h')
        ax6.set_xlabel('Velocidad (km/h)')
        ax6.set_ylabel('Frecuencia')
        ax6.set_title('Distribución de velocidades')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'ANÁLISIS DE CONDUCCIÓN - Secuencia {selected_seq2}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

# Tab 3: Análisis Detallado 
with tab3:
    st.header("📈 Análisis Detallado de Conducción")
    
    selected_seq3 = st.selectbox("Selecciona una secuencia:", available_seqs, key="detail_seq")
    
    if selected_seq3:
        df_seq = df_total[df_total['sequence_id'] == selected_seq3].copy()
        df_seq = df_seq.sort_values('frame_id')
        
        # Preparar datos
        df_seq['timestamp'] = pd.to_datetime(df_seq['frame_id'], unit='ms')
        df_seq['time_seconds'] = (df_seq['timestamp'] - df_seq['timestamp'].iloc[0]).dt.total_seconds()
        df_seq['delta_time'] = df_seq['time_seconds'].diff()
        df_seq['speed_y'] = df_seq['pose_y'].diff() / df_seq['delta_time']
        
        # Cálculos
        df_seq['stopped'] = df_seq['speed_y'].abs() < 0.5
        stopped_time = df_seq[df_seq['stopped']]['delta_time'].sum()
        total_time = df_seq['time_seconds'].iloc[-1]
        
        df_seq['speed_kmh'] = df_seq['speed_y'].abs() * 3.6
        df_seq['speed_category'] = pd.cut(df_seq['speed_kmh'], 
                                         bins=[0, 5, 10, 20, 30, 100],
                                         labels=['Parado (<5)', 'Muy lento (5-10)', 
                                                 'Lento (10-20)', 'Normal (20-30)', 'Rápido (>30)'])
        
        df_seq['acceleration'] = df_seq['speed_y'].diff() / df_seq['delta_time']
        df_seq['heading_change'] = np.arctan2(df_seq['pose_x'].diff(), df_seq['pose_y'].diff())
        df_seq['turning'] = df_seq['heading_change'].abs() > 0.1
        
        # Métricas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("% Paradas", f"{stopped_time/total_time*100:.1f}%")
        with col2:
            st.metric("Aceleración prom", f"{df_seq['acceleration'].mean():.3f} m/s²")
        with col3:
            st.metric("Curvas", f"{df_seq['turning'].sum()}")
        with col4:
            st.metric("Velocidad máx", f"{df_seq['speed_y'].abs().max()*3.6:.1f} km/h")
        
        # Distribución de velocidades
        st.subheader("📊 Distribución de Velocidades")
        speed_dist = df_seq['speed_category'].value_counts(normalize=True) * 100
        
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        bars = ax_dist.bar(speed_dist.index.astype(str), speed_dist.values, 
                          color=['red', 'orange', 'yellow', 'green', 'blue'], edgecolor='black')
        ax_dist.set_xlabel('Categoría de velocidad')
        ax_dist.set_ylabel('Porcentaje (%)')
        ax_dist.set_title('Distribución de velocidades')
        ax_dist.tick_params(axis='x', rotation=45)
        
        for bar, perc in zip(bars, speed_dist.values):
            height = bar.get_height()
            ax_dist.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{perc:.1f}%', ha='center', va='bottom')
        
        st.pyplot(fig_dist)
        
        # Perfil de velocidad
        st.subheader("📈 Perfil de Velocidad vs Tiempo")
        
        fig_vel, ax_vel = plt.subplots(figsize=(12, 6))
        colors = {'Parado (<5)': 'red', 'Muy lento (5-10)': 'orange', 
                 'Lento (10-20)': 'yellow', 'Normal (20-30)': 'green', 'Rápido (>30)': 'blue'}
        
        for cat, color in colors.items():
            mask = df_seq['speed_category'] == cat
            if mask.any():
                ax_vel.scatter(df_seq.loc[mask, 'time_seconds'], 
                             df_seq.loc[mask, 'speed_kmh'],
                             color=color, s=10, alpha=0.5, label=cat)
        
        ax_vel.set_xlabel('Tiempo (s)')
        ax_vel.set_ylabel('Velocidad (km/h)')
        ax_vel.set_title(f'Perfil de velocidad - Secuencia {selected_seq3}')
        ax_vel.legend(markerscale=2)
        ax_vel.grid(True, alpha=0.3)
        
        st.pyplot(fig_vel)

# Tab 4: Comparativas Globales 
with tab4:
    st.header("📋 Comparativas Globales entre Secuencias")
    
    # 1. Distribución de objetos detectados
    st.subheader("🚗 Distribución de Objetos Detectados por Secuencia")
    
    count_cols = [col for col in df_total.columns if col.startswith('count_')]
    if count_cols:
        obj_counts = df_total.groupby('sequence_id')[count_cols].sum()
        obj_counts.columns = [col.replace('count_', '') for col in obj_counts.columns]
        
        fig_obj, ax_obj = plt.subplots(figsize=(14, 6))
        obj_counts.T.plot(kind='bar', ax=ax_obj, edgecolor='black')
        ax_obj.set_xlabel('Tipo de objeto')
        ax_obj.set_ylabel('Cantidad total')
        ax_obj.set_title('Objetos detectados por tipo y secuencia')
        ax_obj.legend(title='Secuencia', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_obj.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig_obj)
        
        # Mostrar tabla
        with st.expander("📊 Ver tabla de objetos"):
            st.dataframe(obj_counts, use_container_width=True)
    
    # 2. Estadísticas comparativas
    st.subheader("📈 Estadísticas Comparativas por Secuencia")
    
    # Calcular estadísticas para cada secuencia
    stats_list = []
    for seq_id, group in df_total.groupby('sequence_id'):
        group = group.copy().sort_values('frame_id')
        group['timestamp'] = pd.to_datetime(group['frame_id'], unit='ms')
        
        # Distancia
        dx = group['pose_x'].iloc[-1] - group['pose_x'].iloc[0] if len(group) > 0 else 0
        dy = group['pose_y'].iloc[-1] - group['pose_y'].iloc[0] if len(group) > 0 else 0
        distancia = np.sqrt(dx**2 + dy**2)
        
        # Duración
        duracion = (group['timestamp'].iloc[-1] - group['timestamp'].iloc[0]).total_seconds() if len(group) > 0 else 0
        
        # Velocidad promedio
        velocidad = (distancia / duracion * 3.6) if duracion > 0 else 0
        
        # Puntos LIDAR promedio
        lidar_prom = group['lidar_n_points_under_100m'].mean() if 'lidar_n_points_under_100m' in group.columns else 0
        
        stats_list.append({
            'Secuencia': seq_id,
            'Frames': len(group),
            'Duración (s)': duracion,
            'Distancia (m)': distancia,
            'Velocidad (km/h)': velocidad,
            'LIDAR pts': lidar_prom
        })
    
    stats_df = pd.DataFrame(stats_list)
    
    # Mostrar tabla
    st.dataframe(stats_df, use_container_width=True)
    
    # Gráficos de barras comparativas
    fig_stats, axes = plt.subplots(2, 3, figsize=(16, 10))
    seq_ids = stats_df['Secuencia'].astype(str)
    
    # 1. Duración
    axes[0,0].bar(seq_ids, stats_df['Duración (s)'], color='skyblue', edgecolor='black')
    axes[0,0].set_ylabel('Duración (s)')
    axes[0,0].set_title('Duración por secuencia')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Distancia
    axes[0,1].bar(seq_ids, stats_df['Distancia (m)'], color='lightgreen', edgecolor='black')
    axes[0,1].set_ylabel('Distancia (m)')
    axes[0,1].set_title('Distancia recorrida')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Velocidad
    axes[0,2].bar(seq_ids, stats_df['Velocidad (km/h)'], color='salmon', edgecolor='black')
    axes[0,2].set_ylabel('Velocidad (km/h)')
    axes[0,2].set_title('Velocidad promedio')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # 4. Frames
    axes[1,0].bar(seq_ids, stats_df['Frames'], color='gold', edgecolor='black')
    axes[1,0].set_ylabel('Número de frames')
    axes[1,0].set_title('Frames por secuencia')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 5. Puntos LIDAR
    axes[1,1].bar(seq_ids, stats_df['LIDAR pts'], color='violet', edgecolor='black')
    axes[1,1].set_ylabel('Puntos LIDAR')
    axes[1,1].set_title('Puntos LIDAR < 100m (promedio)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # 6. Densidad de frames
    axes[1,2].bar(seq_ids, stats_df['Frames'] / stats_df['Duración (s)'], color='orange', edgecolor='black')
    axes[1,2].set_ylabel('Frames/s')
    axes[1,2].set_title('Frecuencia de captura')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.suptitle('ESTADÍSTICAS COMPARATIVAS POR SECUENCIA', fontsize=16, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_stats)

# Tab 5: Limpieza LiDAR
with tab5:
    st.header("🔍 Limpieza y Análisis de Datos LIDAR")
    
    # Función para análisis LIDAR 
    def analyze_lidar_bin_with_filter(file_path, bytes_per_point=16, max_distance=100):
        """
        Analiza un archivo .bin de LIDAR y filtra puntos más allá de max_distance
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            bytes_per_point = 16  # ONCE usa 4 floats
            n_points = len(data) // bytes_per_point
            
            if n_points > 0:
                points = np.frombuffer(data[:n_points * bytes_per_point], dtype=np.float32).reshape(-1, 4)
                
                # Calcular distancia horizontal
                distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
                
                # Filtrar puntos
                mask = distances <= max_distance
                points_filtered = points[mask]
                distances_filtered = distances[mask]
                
                return {
                    'n_points_total': n_points,
                    'n_points_filtered': len(points_filtered),
                    'max_distance': max_distance,
                    'points_original': points,
                    'points_filtered': points_filtered,
                    'distances_filtered': distances_filtered,
                    'distances_all': distances
                }
            return None
            
        except Exception as e:
            st.error(f"Error procesando {file_path}: {e}")
            return None
    
    # Visualización por rangos de distancia
    def visualize_lidar_distance_ranges_streamlit(points, title="LIDAR - Rangos de distancia", max_points=50000):
        """
        Visualiza puntos LIDAR con colores discretos por rangos de distancia - Adaptada para Streamlit
        """
        if len(points) > max_points:
            points = points[:max_points]
        
        # Calcular distancia
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        
        # Definir rangos de distancia y colores
        distance_ranges = [0, 20, 40, 60, 80, 100, 150, float('inf')]
        colors = ['purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
        labels = ['0-20m', '20-40m', '40-60m', '60-80m', '80-100m', '100-150m', '>150m']
        
        # Asignar color según rango
        point_colors = []
        for d in distances:
            for i, (low, high) in enumerate(zip(distance_ranges[:-1], distance_ranges[1:])):
                if low <= d < high:
                    point_colors.append(colors[i])
                    break
        
        fig = plt.figure(figsize=(14, 10))
        
        # 1. Vista 3D
        ax1 = fig.add_subplot(221, projection='3d')
        for i, color in enumerate(colors):
            mask = np.array(point_colors) == color
            if mask.any():
                ax1.scatter(points[mask, 0], points[mask, 1], points[mask, 2], 
                           color=color, s=1, alpha=0.6, label=labels[i])
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D - Rangos de distancia')
        ax1.legend(loc='upper left', fontsize=8)
        
        # 2. Vista superior
        ax2 = fig.add_subplot(222)
        for i, color in enumerate(colors):
            mask = np.array(point_colors) == color
            if mask.any():
                ax2.scatter(points[mask, 0], points[mask, 1], 
                           color=color, s=1, alpha=0.6, label=labels[i])
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Vista superior - Rangos de distancia')
        ax2.set_aspect('equal')
        ax2.legend(loc='upper right', fontsize=8)
        
        # 3. Gráfico de distribución por rangos
        ax3 = fig.add_subplot(223)
        counts = []
        for i, (low, high) in enumerate(zip(distance_ranges[:-1], distance_ranges[1:])):
            count = ((distances >= low) & (distances < high)).sum()
            counts.append(count)
        
        bars = ax3.bar(labels, counts, color=colors, edgecolor='black')
        ax3.set_xlabel('Rango de distancia')
        ax3.set_ylabel('Número de puntos')
        ax3.set_title('Distribución por rangos de distancia')
        ax3.tick_params(axis='x', rotation=45)
        
        # Añadir etiquetas con valores
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=9)
        
        # 4. Gráfico circular
        ax4 = fig.add_subplot(224)
        wedges, texts, autotexts = ax4.pie(counts, colors=colors, autopct='%1.1f%%',
                                           startangle=90, textprops={'fontsize': 9})
        ax4.set_title('Porcentaje por rango de distancia')
        ax4.legend(wedges, labels, title="Rangos", loc="center left", 
                   bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, counts, labels, colors, distances
    
    # Configuración de la página
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 Configuración")
        
        # Buscar archivos en el directorio
        LIDAR_DIR = "./lidar_clean"
        
        if os.path.exists(LIDAR_DIR):
            # Listar archivos .jpg disponibles
            image_files = sorted([f for f in os.listdir(LIDAR_DIR) if f.endswith('.jpg')])
            
            if image_files:
                # Seleccionar archivo
                selected_image = st.selectbox(
                    "Selecciona una imagen:",
                    image_files,
                    index=0
                )
                
                # Obtener el archivo .bin correspondiente
                base_name = selected_image.replace('.jpg', '')
                bin_file = os.path.join(LIDAR_DIR, f"{base_name}.bin")
                img_file = os.path.join(LIDAR_DIR, selected_image)
                
                # Configurar distancia máxima
                max_distance = st.slider(
                    "Distancia máxima (metros):",
                    min_value=10,
                    max_value=200,
                    value=100,
                    step=10
                )
                
                # Botón para analizar
                if st.button("🔍 Analizar LIDAR", type="primary", use_container_width=True):
                    if os.path.exists(bin_file):
                        # Analizar LIDAR
                        with st.spinner("Analizando datos LIDAR..."):
                            result = analyze_lidar_bin_with_filter(bin_file, max_distance=max_distance)
                        
                        if result:
                            st.session_state['lidar_result'] = result
                            st.session_state['current_image'] = img_file
                            st.session_state['current_bin'] = bin_file
                            st.session_state['selected_image_name'] = selected_image
                            st.success("¡Análisis completado!")
                    else:
                        st.error(f"No se encontró el archivo .bin correspondiente: {bin_file}")
                
                # Mostrar estadísticas si ya hay análisis
                if 'lidar_result' in st.session_state:
                    result = st.session_state['lidar_result']
                    
                    st.subheader("📊 Estadísticas")
                    
                    # Métricas en columnas
                    mcol1, mcol2, mcol3 = st.columns(3)
                    
                    with mcol1:
                        st.metric(
                            "Puntos totales",
                            f"{result['n_points_total']:,}",
                            delta=f"{result['n_points_filtered']:,} filtrados"
                        )
                    
                    with mcol2:
                        porcentaje = (result['n_points_filtered'] / result['n_points_total']) * 100
                        st.metric(
                            "Conservados",
                            f"{porcentaje:.1f}%",
                            delta=f"≤ {result['max_distance']}m"
                        )
                    
                    with mcol3:
                        st.metric(
                            "Eliminados",
                            f"{result['n_points_total'] - result['n_points_filtered']:,}",
                            delta="puntos"
                        )
                    
                    # Estadísticas detalladas
                    with st.expander("📈 Ver estadísticas detalladas", expanded=False):
                        if len(result['points_filtered']) > 0:
                            st.write("**Coordenadas de puntos filtrados:**")
                            st.write(f"- X: Min={result['points_filtered'][:, 0].min():.2f}m, Max={result['points_filtered'][:, 0].max():.2f}m")
                            st.write(f"- Y: Min={result['points_filtered'][:, 1].min():.2f}m, Max={result['points_filtered'][:, 1].max():.2f}m")
                            st.write(f"- Z: Min={result['points_filtered'][:, 2].min():.2f}m, Max={result['points_filtered'][:, 2].max():.2f}m")
                            st.write(f"- Intensidad: Promedio={result['points_filtered'][:, 3].mean():.2f}")
                            
                            st.write("\n**Distancias:**")
                            st.write(f"- Mínima: {result['distances_filtered'].min():.2f}m")
                            st.write(f"- Máxima: {result['distances_filtered'].max():.2f}m")
                            st.write(f"- Promedio: {result['distances_filtered'].mean():.2f}m")
                            
                            # Distribución por rangos
                            st.write("\n**Distribución por rangos de distancia:**")
                            bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                            labels = ['0-10m', '10-20m', '20-30m', '30-40m', '40-50m', 
                                     '50-60m', '60-70m', '70-80m', '80-90m', '90-100m']
                            
                            hist, _ = np.histogram(result['distances_filtered'], bins=bins)
                            for label, count in zip(labels, hist):
                                if count > 0:
                                    percentage = count / len(result['distances_filtered']) * 100
                                    st.write(f"- {label}: {count:,} puntos ({percentage:.1f}%)")
            else:
                st.warning(f"No se encontraron archivos .jpg en {LIDAR_DIR}")
        else:
            st.error(f"Directorio no encontrado: {LIDAR_DIR}")
            st.info("Crea una carpeta 'lidar_clean' con archivos .jpg y .bin")
    
    with col2:
        st.subheader("📊 Visualización")
        
        if 'current_image' in st.session_state and os.path.exists(st.session_state['current_image']):
            try:
                image = Image.open(st.session_state['current_image'])
                st.image(image, caption="Imagen correspondiente", use_column_width=True)
            except:
                st.warning("No se pudo cargar la imagen")
        
        # gráficos LIDAR si hay análisis
        if 'lidar_result' in st.session_state:
            result = st.session_state['lidar_result']
            
            # Sub-tabs para diferentes visualizaciones
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                "Comparación", 
                "Distribución", 
                "3D", 
                "Rangos de Distancia"  
            ])
            
            with viz_tab1:
                fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Comparación antes/después
                ax1.scatter(result['points_original'][:, 0], result['points_original'][:, 1], 
                          s=0.1, alpha=0.3, label=f"Original ({result['n_points_total']:,})", color='blue')
                ax1.scatter(result['points_filtered'][:, 0], result['points_filtered'][:, 1], 
                          s=0.5, alpha=0.7, label=f"Filtrado ({result['n_points_filtered']:,})", color='red')
                ax1.set_xlabel('X (m)')
                ax1.set_ylabel('Y (m)')
                ax1.set_title('Comparación antes/después del filtrado')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_aspect('equal')
                
                # Distribución de distancias
                ax2.hist(result['distances_all'], bins=50, alpha=0.5, label='Todos los puntos', color='blue')
                ax2.hist(result['distances_filtered'], bins=50, alpha=0.7, 
                        label=f'≤ {result["max_distance"]}m', color='red')
                ax2.axvline(x=result['max_distance'], color='black', linestyle='--', 
                          label=f'Límite {result["max_distance"]}m')
                ax2.set_xlabel('Distancia (m)')
                ax2.set_ylabel('Número de puntos')
                ax2.set_title('Distribución de distancias')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig1)
            
            with viz_tab2:
                fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Puntos filtrados con color por distancia
                scatter1 = ax1.scatter(result['points_filtered'][:, 0], result['points_filtered'][:, 1], 
                                     c=result['distances_filtered'], cmap='viridis', s=1, alpha=0.7)
                ax1.set_xlabel('X (m)')
                ax1.set_ylabel('Y (m)')
                ax1.set_title(f'Puntos filtrados (≤ {result["max_distance"]}m)')
                ax1.set_aspect('equal')
                plt.colorbar(scatter1, ax=ax1, label='Distancia (m)')
                
                # Intensidad vs Distancia
                scatter2 = ax2.scatter(result['distances_filtered'], 
                                     result['points_filtered'][:, 3],  # Intensidad
                                     c=result['points_filtered'][:, 2],  # Color por altura Z
                                     cmap='plasma', s=1, alpha=0.7)
                ax2.set_xlabel('Distancia (m)')
                ax2.set_ylabel('Intensidad')
                ax2.set_title('Intensidad vs Distancia (color por altura Z)')
                plt.colorbar(scatter2, ax=ax2, label='Altura Z (m)')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig2)
            
            with viz_tab3:
                # Vista 3D
                fig3 = plt.figure(figsize=(10, 8))
                ax = fig3.add_subplot(111, projection='3d')
                
                scatter = ax.scatter(result['points_filtered'][:, 0], 
                                   result['points_filtered'][:, 1], 
                                   result['points_filtered'][:, 2],
                                   c=result['distances_filtered'],
                                   cmap='viridis', s=1, alpha=0.7)
                
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.set_title(f'Vista 3D - Puntos filtrados (≤ {result["max_distance"]}m)')
                plt.colorbar(scatter, ax=ax, label='Distancia (m)')
                
                plt.tight_layout()
                st.pyplot(fig3)
            
            # NUEVA TAB: Rangos de Distancia
            with viz_tab4:
                st.subheader("📊 Análisis por Rangos de Distancia")
                
                # Usar la función de visualización por rangos
                if 'selected_image_name' in st.session_state:
                    title = f"LIDAR: {st.session_state['selected_image_name'].replace('.jpg', '')}"
                    
                    # Limitar puntos si es necesario para mejor rendimiento
                    max_points_to_show = min(50000, len(result['points_filtered']))
                    points_sample = result['points_filtered'][:max_points_to_show]
                    
                    # Generar visualización
                    fig_rangos, counts, labels_rangos, colors_rangos, distances_rangos = visualize_lidar_distance_ranges_streamlit(
                        points_sample, 
                        title=title
                    )
                    
                    # Mostrar el gráfico
                    st.pyplot(fig_rangos)
                    
                    # Mostrar estadísticas adicionales
                    with st.expander("📈 Ver estadísticas detalladas por rango", expanded=True):
                        total_points = len(points_sample)
                        st.write("**Distribución por rangos de distancia:**")
                        
                        # Crear tabla de estadísticas
                        stats_data = []
                        for i, (label, count, color) in enumerate(zip(labels_rangos, counts, colors_rangos)):
                            percentage = (count / total_points) * 100 if total_points > 0 else 0
                            stats_data.append({
                                'Rango': label,
                                'Color': color,
                                'Puntos': f"{count:,}",
                                'Porcentaje': f"{percentage:.1f}%"
                            })
                        
                        # Mostrar tabla
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        
                        # Métricas adicionales
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Puntos analizados", f"{total_points:,}")
                        with col2:
                            st.metric("Distancia promedio", f"{distances_rangos.mean():.1f}m")
                        with col3:
                            st.metric("Distancia máxima", f"{distances_rangos.max():.1f}m")
            
            # Información del archivo actual
            with st.expander("📄 Información del archivo", expanded=False):
                if 'current_bin' in st.session_state:
                    file_path = st.session_state['current_bin']
                    file_size = os.path.getsize(file_path)
                    st.write(f"**Archivo:** {os.path.basename(file_path)}")
                    st.write(f"**Tamaño:** {file_size:,} bytes")
                    st.write(f"**Formato:** 4 valores por punto (x, y, z, intensity)")
                    st.write(f"**Puntos por archivo:** {result['n_points_total']:,}")
                    st.write(f"**Puntos conservados:** {result['n_points_filtered']:,}")
                    st.write(f"**Distancia máxima:** {result['max_distance']}m")
        
        else:
            st.info("Selecciona una imagen y haz clic en 'Analizar LIDAR' para ver los resultados")
    
    # Información adicional
    with st.expander("ℹ️ Información sobre el filtrado LIDAR", expanded=False):
        st.write("""
        ### ¿Qué hace este análisis?
        
        1. **Carga datos LIDAR**: Lee archivos .bin con coordenadas (x, y, z) e intensidad
        2. **Calcula distancias**: Distancia horizontal desde el vehículo: √(x² + y²)
        3. **Filtra puntos**: Elimina puntos más allá de la distancia máxima configurada
        4. **Analiza estadísticas**: Calcula distribuciones, rangos y métricas
        
        ### ¿Por qué filtrar puntos lejanos?
        - **Reducción de ruido**: Puntos muy lejanos suelen tener más error
        - **Enfoque en lo relevante**: Para vehículos autónomos, lo cercano es más importante
        - **Optimización**: Menos datos para procesar = más rápido
        
        ### Formato de datos LIDAR:
        - **4 valores por punto**: x, y, z (metros), intensity (0-255)
        - **Coordenadas relativas**: Origen en el vehículo
        - **Unidad**: 16 bytes por punto (4 floats × 4 bytes)
        
        ### Análisis por Rangos
        La nueva visualización muestra los puntos agrupados por rangos de distancia:
        - **Colores diferenciados**: Cada rango tiene un color único
        - **Vista 3D y 2D**: Para mejor comprensión espacial
        - **Gráficos de distribución**: Barras y pastel para entender proporciones
        - **Estadísticas detalladas**: Tabla con conteos y porcentajes por rango
        """)

# Tab ENTRENAR MODELOS
with tab6:
    train_models_tab()

# Tab: BENTOML PREDICTOR 
with tab7:
    bento_tab()