"""
Flask 메인 애플리케이션
시계열 데이터 분석 및 예측 웹 애플리케이션
"""

import os
import warnings
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# Flask 앱 생성
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 실제 운영에서는 환경변수로 설정

# 업로드 설정
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한

# 업로드 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fix_24_hour(time_str):
    """24:00 시간 형식 수정"""
    s = str(time_str).strip()
    if ' 24:00' in s or s.endswith('24:00:00'):
        date_part = s.split(' ')[0]
        new_date = datetime.strptime(date_part, '%Y/%m/%d') + timedelta(days=1)
        has_seconds = s.endswith('24:00:00')
        return new_date.strftime('%Y/%m/%d') + (' 00:00:00' if has_seconds else ' 00:00')
    return s

@app.route('/')
def home():
    """홈페이지"""
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """파일 업로드 처리"""
    if 'file' not in request.files:
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 데이터 로드
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            
            # 날짜 형식 컬럼 확인 후 변환
            for column in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    df[column] = df[column].apply(fix_24_hour)
                    df.rename(columns={column: 'logTime'}, inplace=True)
                    break
                if pd.api.types.is_string_dtype(df[column]):
                    try:
                        df[column] = df[column].apply(fix_24_hour)
                        df[column] = pd.to_datetime(df[column], errors='coerce', infer_datetime_format=True)
                        df.rename(columns={column: 'logTime'}, inplace=True)
                        break
                    except ValueError:
                        continue
            
            if 'logTime' not in df.columns:
                return jsonify({'error': '날짜 형식의 컬럼을 찾을 수 없습니다.'}), 400
            
            # 세션에 데이터 저장 (실제로는 데이터베이스나 캐시 사용 권장)
            session['df'] = df.to_json(orient='records', date_format='iso')
            session['columns'] = list(df.columns)
            session['filename'] = filename
            
            # 데이터 요약 정보
            summary = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'start_date': str(df['logTime'].min()),
                'end_date': str(df['logTime'].max())
            }
            
            return jsonify({
                'success': True,
                'message': '파일이 성공적으로 업로드되었습니다.',
                'summary': summary
            })
            
        except Exception as e:
            return jsonify({'error': f'파일 처리 중 오류가 발생했습니다: {str(e)}'}), 500
    
    return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

@app.route('/data_summary')
def data_summary():
    """데이터 요약 페이지"""
    if 'df' not in session:
        return redirect(url_for('home'))
    
    df = pd.read_json(session['df'], orient='records')
    df['logTime'] = pd.to_datetime(df['logTime'])
    
    return render_template('data_summary.html', 
                         columns=session['columns'],
                         filename=session['filename'])

@app.route('/data_analysis')
def data_analysis():
    """데이터 분석 페이지"""
    if 'df' not in session:
        return redirect(url_for('home'))
    
    return render_template('data_analysis.html')

@app.route('/model_training')
def model_training():
    """모델 훈련 페이지"""
    if 'df' not in session:
        return redirect(url_for('home'))
    
    return render_template('model_training.html')

@app.route('/prediction')
def prediction():
    """예측 페이지"""
    if 'df' not in session:
        return redirect(url_for('home'))
    
    return render_template('prediction.html')

# API 엔드포인트들
@app.route('/api/data/columns')
def api_get_columns():
    """컬럼 목록 API"""
    if 'columns' not in session:
        return jsonify({'error': '데이터가 없습니다.'}), 400
    
    return jsonify({'columns': session['columns']})

@app.route('/api/data/summary')
def api_get_summary():
    """데이터 요약 API"""
    if 'df' not in session:
        return jsonify({'error': '데이터가 없습니다.'}), 400
    
    df = pd.read_json(session['df'], orient='records')
    df['logTime'] = pd.to_datetime(df['logTime'])
    
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': df.isnull().sum().to_dict(),
        'start_date': str(df['logTime'].min()),
        'end_date': str(df['logTime'].max()),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist()
    }
    
    return jsonify(summary)

@app.route('/api/data/preview')
def api_get_preview():
    """데이터 미리보기 API"""
    if 'df' not in session:
        return jsonify({'error': '데이터가 없습니다.'}), 400
    
    df = pd.read_json(session['df'], orient='records')
    df['logTime'] = pd.to_datetime(df['logTime'])
    
    # 처음 100행만 반환
    preview = df.head(100).to_dict('records')
    
    return jsonify({'data': preview})

@app.route('/api/analysis/outliers', methods=['POST'])
def api_analyze_outliers():
    """이상치 분석 API"""
    if 'df' not in session:
        return jsonify({'error': '데이터가 없습니다.'}), 400
    
    data = request.get_json()
    target_column = data.get('target_column')
    
    if not target_column:
        return jsonify({'error': '대상 컬럼이 지정되지 않았습니다.'}), 400
    
    try:
        df = pd.read_json(session['df'], orient='records')
        df['logTime'] = pd.to_datetime(df['logTime'])
        
        # 이상치 분석 로직 (기존 utils/data_processor.py의 함수 활용)
        from utils.data_processor import cached_analyze_outliers
        
        series = df.set_index('logTime')[target_column]
        outliers = cached_analyze_outliers(series)
        
        return jsonify({'outliers': outliers})
        
    except Exception as e:
        return jsonify({'error': f'이상치 분석 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/api/model/train', methods=['POST'])
def api_train_model():
    """모델 훈련 API"""
    if 'df' not in session:
        return jsonify({'error': '데이터가 없습니다.'}), 400
    
    data = request.get_json()
    target_column = data.get('target_column')
    models = data.get('models', [])
    strategy = data.get('strategy', 'smart')
    
    if not target_column:
        return jsonify({'error': '대상 컬럼이 지정되지 않았습니다.'}), 400
    
    try:
        # 모델 훈련 로직 (기존 backend/model_service.py의 함수 활용)
        # 여기서는 간단한 응답만 반환
        return jsonify({
            'success': True,
            'message': '모델 훈련이 시작되었습니다.',
            'target_column': target_column,
            'models': models,
            'strategy': strategy
        })
        
    except Exception as e:
        return jsonify({'error': f'모델 훈련 중 오류가 발생했습니다: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)



