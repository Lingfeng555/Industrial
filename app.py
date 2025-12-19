from unittest import result
import streamlit as st
import sys
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage
import torch
import requests
from args import args
from io import BytesIO
import bentoml
import time
import base64

from src.once_dataset import ONCEDataset
from car_simulator import Car

# Import processing functions
import sys
sys.path.insert(0, './processing')
from processing.app_final import *

available_cars = ["000027","000028", "000112", "000201"]
cars = {}
for id in available_cars:
    cars[id] = Car(car_id=id, data_path=args.data_path)

cams = cars[id].cams
client = bentoml.SyncHTTPClient('http://localhost:3000')
image_server_client = bentoml.SyncHTTPClient('http://localhost:3001')
speed_prediction_client = bentoml.SyncHTTPClient('http://localhost:3002')
# Define inference function
def yolo_api_inference(image: torch.Tensor, confidence: float) -> np.ndarray:
    img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    _, buffer = cv2.imencode('.jpg', img_bgr)
    img_base64_input = base64.b64encode(buffer).decode('utf-8')

    print("Sending inference request to YOLO service...")
    
    response = client.yolo_inference(image_b64=img_base64_input, confidence=confidence)
    
    entities = response["pred_entities"]
    
    img_data = base64.b64decode(response["image_base64"])
    bbbox_img = np.array(Image.open(BytesIO(img_data)))
    
    print(bbbox_img.shape)
    return entities, bbbox_img

# Streamlit App
if "cam_busy" not in st.session_state:
    st.session_state.cam_busy = False

if "current_image_index" not in st.session_state:
    st.session_state.current_image_index = 0
    
st.title("🚗 Driver Monitor")
st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Dataset Overview", 
    "Camera Views", 
    "LiDAR Point Clouds", 
    "Sensor Fusion", 
    "Data Statistics", 
    "Model Training", 
    "Evaluation Metrics",
    "Real-Time Inference",
    "Saved Inference Frames"
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
            st.dataframe(strong_corr_df, width = "stretch")
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
            st.dataframe(obj_counts, width = "stretch")
    
    # 2. Estadísticas comparativas
    st.subheader("📈 Estadísticas Comparativas por Secuencia")
    width='content'
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
    st.dataframe(stats_df, width = "stretch")
    
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
                if st.button("🔍 Analizar LIDAR", type="primary", width = "stretch"):
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
                        st.dataframe(stats_df, width = "stretch", hide_index=True)
                        
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

with tab8:
    selected_car = st.selectbox("Select Car", options=available_cars, index=0)
    selected_cam = st.selectbox("Select Camera", options=cars[selected_car].cams, index=0)
    selected_confidence = st.slider("Select Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

    #Read in "real time"
    @st.fragment(run_every=0.1)
    def show_car_cam():
            
        start = time.time()
        car = cars[selected_car]
        lidar_data, images = car.get_info()
        image = images[selected_cam]

        print("Inference started")
        start_inference = time.time()
        entities, bbbox_img = yolo_api_inference(image, selected_confidence)
        print("Inference completed")
        print(f"Inference time: {time.time() - start_inference:.3f}s")

        st.image(bbbox_img, caption=f"From car {selected_car}") 
        st.write(f"Detected {len(entities)} entities in the selected frame.")
        st.write(f"Frame time: {time.time() - start:.3f}s")

        st.session_state.cam_busy = False

    show_car_cam()

with tab9:
    st.subheader("Saved Inference Frames")
    try:
        response = image_server_client.list_images()
        images_list = response.get("images", [])
        
        if not images_list:
            st.info("No images found in inference_outputs directory")
        else:
            st.write(f"Found {len(images_list)} saved frames")
            
            
            
            # Image display with auto-play
            @st.fragment(run_every=0.5)
            def show_image_sequence():
                if not images_list:
                    return
                
                st.session_state.current_image_index = (st.session_state.current_image_index + 1) % len(images_list)
                
                # Ensure index is within bounds
                if st.session_state.current_image_index >= len(images_list):
                    st.session_state.current_image_index = 0
                
                current_image = images_list[st.session_state.current_image_index]
                
                # Fetch and display the current image
                img_response = image_server_client.get_image(filename=current_image)
                
                if "error" in img_response:
                    st.error(f"Error loading image: {img_response['error']}")
                else:
                    img_data = base64.b64decode(img_response["image_base64"])
                    img = Image.open(BytesIO(img_data))
                    st.image(img, caption=f"{current_image} ({st.session_state.current_image_index + 1}/{len(images_list)})", width='stretch')
                    
            show_image_sequence()
            
            new_index = st.slider(
                "Frame", 
                min_value=0, 
                max_value=len(images_list) - 1, 
                value=st.session_state.current_image_index,
                key="frame_slider"
            )
            if new_index != st.session_state.current_image_index:
                st.session_state.current_image_index = new_index
                    
    except Exception as e:
        st.error(f"Could not connect to Image Server Service: {str(e)}")
        st.info("Make sure the ImageServerService is running on port 3001")