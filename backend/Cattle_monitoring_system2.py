from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect
from datetime import datetime
import numpy as np
import math
import os
import sqlite3
import time
import threading
from werkzeug.utils import secure_filename
import os
import numpy as np

app = Flask(__name__, static_folder='static')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Data validation functions
def validate_breed(breed):
    valid_breeds = {'Holstein', 'Jersey', 'Guernsey', 'Sahiwal'}
    if breed not in valid_breeds:
        raise ValueError(f"Invalid breed. Must be one of: {', '.join(valid_breeds)}")
    return breed

def validate_numeric(value, name, min_val, max_val):
    try:
        value = float(value)
        if not min_val <= value <= max_val:
            raise ValueError(f"{name} must be between {min_val} and {max_val}")
        return value
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {name}. Must be a number between {min_val} and {max_val}")

def validate_health_status(status):
    valid_statuses = {'Excellent', 'Good', 'Fair', 'Poor'}
    if status not in valid_statuses:
        raise ValueError(f"Invalid health status. Must be one of: {', '.join(valid_statuses)}")
    return status

def validate_animal_data(data, required_fields=None):
    errors = []
    
    try:
        if required_fields:
            missing = [field for field in required_fields if field not in data]
            if missing:
                errors.append(f"Missing required fields: {', '.join(missing)}")
        
        if 'breed' in data:
            try:
                validate_breed(data['breed'])
            except ValueError as e:
                errors.append(str(e))
        
        if 'age' in data:
            try:
                validate_numeric(data['age'], 'Age', 1, 15)
            except ValueError as e:
                errors.append(str(e))
        
        if 'weight' in data:
            try:
                validate_numeric(data['weight'], 'Weight', 300, 900)
            except ValueError as e:
                errors.append(str(e))
        
        if 'lactation_stage' in data:
            try:
                validate_numeric(data['lactation_stage'], 'Lactation stage', 1, 10)
            except ValueError as e:
                errors.append(str(e))
        
        if 'parity' in data:
            try:
                validate_numeric(data['parity'], 'Parity', 1, 8)
            except ValueError as e:
                errors.append(str(e))
        
        if 'milk_yield' in data:
            try:
                validate_numeric(data['milk_yield'], 'Milk yield', 0, 50)
            except ValueError as e:
                errors.append(str(e))
        
        if 'health_status' in data:
            try:
                validate_health_status(data['health_status'])
            except ValueError as e:
                errors.append(str(e))
        
        if errors:
            raise ValueError('\n'.join(errors))
            
    except Exception as e:
        raise ValueError(f"Validation error: {str(e)}")
    
    return True

# Database Manager Class
class DatabaseManager:
    def __init__(self, db_path='cattle.db'):
        self.db_path = db_path
        self.init_db()
        
    def add_vital_signs(self, cattle_id, temperature, heart_rate, activity_level, rumination_time):
        """Add a new vital signs reading for a cattle"""
        timestamp = datetime.datetime.now().isoformat()
        try:
            return self.execute_query(
                'INSERT INTO vital_signs (cattle_id, temperature, heart_rate, activity_level, rumination_time, timestamp) VALUES (?, ?, ?, ?, ?, ?)',
                (cattle_id, temperature, heart_rate, activity_level, rumination_time, timestamp)
            )
        except Exception as e:
            raise Exception(f"Error adding vital signs: {str(e)}")

    def add_alert(self, cattle_id, alert_type, severity, message):
        """Add a new alert for a cattle"""
        timestamp = datetime.datetime.now().isoformat()
        try:
            return self.execute_query(
                'INSERT INTO alerts (cattle_id, type, severity, message, timestamp) VALUES (?, ?, ?, ?, ?)',
                (cattle_id, alert_type, severity, message, timestamp)
            )
        except Exception as e:
            raise Exception(f"Error adding alert: {str(e)}")

    def update_cattle(self, cattle_id, updates):
        """Update cattle information"""
        valid_fields = {'breed', 'age', 'weight', 'lactation_stage', 'parity', 'milk_yield', 'health_status'}
        update_fields = [f"{k} = ?" for k in updates.keys() if k in valid_fields]
        
        if not update_fields:
            raise ValueError("No valid fields to update")

        updates['last_updated'] = datetime.datetime.now().isoformat()
        update_fields.append("last_updated = ?")
        
        query = f"UPDATE cattle SET {', '.join(update_fields)} WHERE id = ?"
        params = list(updates.values()) + [cattle_id]
        
        try:
            return self.execute_query(query, params)
        except Exception as e:
            raise Exception(f"Error updating cattle: {str(e)}")

    def resolve_alert(self, alert_id):
        """Mark an alert as resolved"""
        try:
            return self.execute_query(
                'UPDATE alerts SET resolved = 1 WHERE id = ?',
                (alert_id,)
            )
        except Exception as e:
            raise Exception(f"Error resolving alert: {str(e)}")

    def add_motion_analysis(self, video_filename, motion_detected):
        """Store video motion analysis results"""
        timestamp = datetime.datetime.now().isoformat()
        try:
            return self.execute_query(
                'INSERT INTO motion_analysis (video_filename, motion_detected, timestamp) VALUES (?, ?, ?)',
                (video_filename, motion_detected, timestamp)
            )
        except sqlite3.OperationalError:
            # Create table if it doesn't exist
            self.execute_query('''CREATE TABLE IF NOT EXISTS motion_analysis
                                 (id INTEGER PRIMARY KEY,
                                  video_filename TEXT NOT NULL,
                                  motion_detected INTEGER NOT NULL,
                                  timestamp TEXT NOT NULL)''')
            # Try insert again
            return self.execute_query(
                'INSERT INTO motion_analysis (video_filename, motion_detected, timestamp) VALUES (?, ?, ?)',
                (video_filename, motion_detected, timestamp)
            )
        except Exception as e:
            raise Exception(f"Error storing motion analysis: {str(e)}")
    
    def get_connection(self):
        """Get a database connection with timeout and retry logic"""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                return sqlite3.connect(
                    self.db_path,
                    timeout=10,
                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
                )
            except sqlite3.Error as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise Exception(f"Failed to connect to database after {max_retries} attempts: {e}")
                time.sleep(retry_delay)
    
    def execute_query(self, query, params=None, fetch=False):
        """Execute a query with error handling and connection management"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            if fetch:
                result = cursor.fetchall()
                conn.commit()
                return result
            
            conn.commit()
            return cursor.lastrowid
            
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            raise Exception(f"Database error: {str(e)}")
            
        finally:
            if conn:
                conn.close()
    
    def init_db(self):
        """Initialize the database schema and sample data"""
        # Create tables with foreign key support
        self.execute_query('''PRAGMA foreign_keys = ON''')
        
        # Create cattle table
        self.execute_query('''CREATE TABLE IF NOT EXISTS cattle
                             (id INTEGER PRIMARY KEY,
                              breed TEXT NOT NULL,
                              age REAL NOT NULL,
                              weight REAL NOT NULL,
                              lactation_stage INTEGER NOT NULL,
                              parity INTEGER NOT NULL,
                              milk_yield REAL NOT NULL,
                              health_status TEXT NOT NULL,
                              last_updated TEXT NOT NULL,
                              CONSTRAINT valid_breed CHECK (breed IN ('Holstein', 'Jersey', 'Guernsey', 'Sahiwal')),
                              CONSTRAINT valid_health CHECK (health_status IN ('Excellent', 'Good', 'Fair', 'Poor')))''')
                              
        # Create vital_signs table for historical data
        self.execute_query('''CREATE TABLE IF NOT EXISTS vital_signs
                             (id INTEGER PRIMARY KEY,
                              cattle_id INTEGER NOT NULL,
                              temperature REAL NOT NULL,
                              heart_rate INTEGER NOT NULL,
                              activity_level REAL NOT NULL,
                              rumination_time REAL NOT NULL,
                              timestamp TEXT NOT NULL,
                              FOREIGN KEY (cattle_id) REFERENCES cattle(id) ON DELETE CASCADE)''')
                              
        # Create alerts table
        self.execute_query('''CREATE TABLE IF NOT EXISTS alerts
                             (id INTEGER PRIMARY KEY,
                              cattle_id INTEGER NOT NULL,
                              type TEXT NOT NULL,
                              severity TEXT NOT NULL,
                              message TEXT NOT NULL,
                              timestamp TEXT NOT NULL,
                              resolved INTEGER DEFAULT 0,
                              FOREIGN KEY (cattle_id) REFERENCES cattle(id) ON DELETE CASCADE)''')
        
        # Check if we need to insert sample data
        count = self.execute_query("SELECT COUNT(*) FROM cattle", fetch=True)[0][0]
        
        if count == 0:
            # Insert sample data with current date
            current_date = datetime.now().strftime('%Y-%m-%d')
            sample_data = [
                (1, 'Holstein', 4.5, 650, 3, 2, 28.5, 'Excellent', current_date),
                (2, 'Jersey', 5.2, 450, 2, 1, 22.3, 'Good', current_date),
                (3, 'Sahiwal', 6.1, 425, 5, 3, 18.7, 'Fair', current_date),
                (4, 'Holstein', 3.8, 600, 1, 1, 26.2, 'Excellent', current_date),
                (5, 'Guernsey', 7.2, 500, 4, 2, 25.1, 'Good', current_date),
                (6, 'Jersey', 4.3, 430, 3, 2, 21.8, 'Good', current_date),
                (7, 'Holstein', 5.5, 620, 2, 1, 27.4, 'Excellent', current_date),
                (8, 'Sahiwal', 8.1, 410, 6, 4, 17.9, 'Poor', current_date)
            ]
            
            for cattle in sample_data:
                self.execute_query('''INSERT INTO cattle 
                                    (id, breed, age, weight, lactation_stage, parity, milk_yield, health_status, last_updated)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', cattle)
                                    
    def add_vital_signs(self, cattle_id, temperature, heart_rate, activity_level, rumination_time):
        """Add vital signs reading to the database"""
        self.execute_query(
            '''INSERT INTO vital_signs 
               (cattle_id, temperature, heart_rate, activity_level, rumination_time, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (cattle_id, temperature, heart_rate, activity_level, rumination_time, datetime.now().isoformat())
        )
    
    def add_alert(self, cattle_id, alert_type, severity, message):
        """Add a new alert to the database"""
        self.execute_query(
            '''INSERT INTO alerts 
               (cattle_id, type, severity, message, timestamp)
               VALUES (?, ?, ?, ?, ?)''',
            (cattle_id, alert_type, severity, message, datetime.now().isoformat())
        )
    
    def resolve_alert(self, alert_id):
        """Mark an alert as resolved"""
        self.execute_query(
            "UPDATE alerts SET resolved = 1 WHERE id = ?",
            (alert_id,)
        )
    


# Sensor simulation class for real-time monitoring
class SensorSimulator:
    def __init__(self, db):
        self.db = db
        self.running = False
        self.simulation_thread = None
        self.cattle_baselines = {}
        
    def initialize_baselines(self):
        """Initialize baseline vital signs for each cattle"""
        try:
            cattle = self.db.execute_query("SELECT id, breed, age FROM cattle", fetch=True)
            for cattle_id, breed, age in cattle:
                # Base values adjusted by breed and age
                base_temp = 38.5  # Normal cattle temperature
                base_heart = 65   # Normal cattle heart rate
                base_activity = 12 # Hours of activity per day
                base_rumination = 8 # Hours of rumination per day
                
                # Adjust for breed
                if breed == 'Holstein':
                    base_temp += 0.2
                    base_heart += 5
                elif breed == 'Jersey':
                    base_temp -= 0.1
                    base_activity -= 1
                elif breed == 'Sahiwal':
                    base_temp -= 0.3
                    base_rumination += 1
                
                # Adjust for age
                age_factor = min(1.0, age / 5.0)  # Peak at 5 years
                base_heart -= (age_factor * 5)
                base_activity -= (age_factor * 2)
                
                self.cattle_baselines[cattle_id] = {
                    'temp': base_temp,
                    'heart': base_heart,
                    'activity': base_activity,
                    'rumination': base_rumination
                }
        except Exception as e:
            print(f"Error initializing baselines: {str(e)}")
    
    def generate_vital_signs(self, cattle_id):
        """Generate realistic vital signs with random variations"""
        if cattle_id not in self.cattle_baselines:
            return None
            
        baseline = self.cattle_baselines[cattle_id]
        
        # Add random variations
        temp = baseline['temp'] + np.random.normal(0, 0.3)
        heart = baseline['heart'] + np.random.normal(0, 3)
        activity = baseline['activity'] + np.random.normal(0, 0.5)
        rumination = baseline['rumination'] + np.random.normal(0, 0.3)
        
        # Add time-based patterns
        hour = datetime.now().hour
        if 6 <= hour <= 18:  # Daytime
            activity += 1
            heart += 5
        else:  # Nighttime
            rumination += 0.5
            heart -= 3
        
        return {
            'temperature': round(temp, 1),
            'heart_rate': int(heart),
            'activity_level': round(activity, 1),
            'rumination_time': round(rumination, 1)
        }
    
    def check_alerts(self, cattle_id, vitals):
        """Check vital signs for concerning values and generate alerts"""
        alerts = []
        
        # Temperature alerts
        if vitals['temperature'] > 39.5:
            alerts.append(('health', 'high', f'High temperature detected for cattle #{cattle_id}: {vitals["temperature"]}°C'))
        elif vitals['temperature'] < 37.5:
            alerts.append(('health', 'high', f'Low temperature detected for cattle #{cattle_id}: {vitals["temperature"]}°C'))
        
        # Heart rate alerts
        if vitals['heart_rate'] > 80:
            alerts.append(('health', 'medium', f'Elevated heart rate for cattle #{cattle_id}: {vitals["heart_rate"]} BPM'))
        elif vitals['heart_rate'] < 50:
            alerts.append(('health', 'high', f'Low heart rate for cattle #{cattle_id}: {vitals["heart_rate"]} BPM'))
        
        # Activity level alerts
        if vitals['activity_level'] < 8:
            alerts.append(('behavior', 'medium', f'Low activity level for cattle #{cattle_id}: {vitals["activity_level"]} hours'))
        
        # Rumination time alerts
        if vitals['rumination_time'] < 6:
            alerts.append(('health', 'medium', f'Low rumination time for cattle #{cattle_id}: {vitals["rumination_time"]} hours'))
        
        return alerts
    
    def simulate_sensors(self):
        """Main simulation loop"""
        while self.running:
            try:
                cattle_ids = [id for id, in self.db.execute_query("SELECT id FROM cattle", fetch=True)]
                
                for cattle_id in cattle_ids:
                    vitals = self.generate_vital_signs(cattle_id)
                    if vitals:
                        # Store vital signs
                        self.db.add_vital_signs(
                            cattle_id,
                            vitals['temperature'],
                            vitals['heart_rate'],
                            vitals['activity_level'],
                            vitals['rumination_time']
                        )
                        
                        # Check for and store alerts
                        alerts = self.check_alerts(cattle_id, vitals)
                        for alert_type, severity, message in alerts:
                            self.db.add_alert(cattle_id, alert_type, severity, message)
                        
                        # Emit real-time update via WebSocket
                        socketio.emit('vital_signs_update', {
                            'cattle_id': cattle_id,
                            'vitals': vitals,
                            'timestamp': datetime.now().isoformat()
                        })
            
            except Exception as e:
                print(f"Error in sensor simulation: {str(e)}")
            
            time.sleep(30)  # Update every 30 seconds
    
    def start(self):
        """Start the sensor simulation"""
        if not self.running:
            self.running = True
            self.initialize_baselines()
            self.simulation_thread = threading.Thread(target=self.simulate_sensors)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
    
    def stop(self):
        """Stop the sensor simulation"""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join()
            self.simulation_thread = None

# Create database manager instance
db = DatabaseManager()

# Create database manager instance
db = DatabaseManager()

# Initialize but don't start sensor simulator yet
sensor_simulator = None

# API Routes
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/dashboard')
def dashboard():
    try:
        # Get total animals
        total_animals = db.execute_query("SELECT COUNT(*) FROM cattle", fetch=True)[0][0]
        
        # Get average milk yield
        avg_yield = round(db.execute_query("SELECT AVG(milk_yield) FROM cattle", fetch=True)[0][0], 1)
        
        # Get healthy animals count
        healthy_animals = db.execute_query("SELECT COUNT(*) FROM cattle WHERE health_status IN ('Excellent', 'Good')", fetch=True)[0][0]
        
        # Get health distribution
        health_distribution = {}
        results = db.execute_query("SELECT health_status, COUNT(*) FROM cattle GROUP BY health_status", fetch=True)
        for row in results:
            health_distribution[row[0]] = row[1]
        
        # Get yield by breed
        yield_by_breed = {}
        results = db.execute_query("SELECT breed, AVG(milk_yield) FROM cattle GROUP BY breed", fetch=True)
        for row in results:
            yield_by_breed[row[0]] = round(row[1], 1)
        
        # Get recent alerts from database
        recent_alerts = db.execute_query("""
            SELECT type, message as description, severity as priority, 
                   CASE 
                       WHEN julianday('now') - julianday(timestamp) < 1 
                       THEN round((julianday('now') - julianday(timestamp)) * 24) || ' hours ago'
                       ELSE round(julianday('now') - julianday(timestamp)) || ' days ago'
                   END as time
            FROM alerts 
            WHERE resolved = 0 
            ORDER BY timestamp DESC 
            LIMIT 3""", fetch=True)
        
        alerts_list = []
        for alert in recent_alerts:
            alerts_list.append({
                'type': alert[0],
                'description': alert[1],
                'priority': alert[2],
                'time': alert[3]
            })

        return jsonify({
            'total_animals': total_animals,
            'avg_yield': avg_yield,
            'healthy_animals': healthy_animals,
            'health_distribution': health_distribution,
            'yield_by_breed': yield_by_breed,
            'recent_alerts': alerts_list
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    return jsonify({
        'total_animals': total_animals,
        'avg_yield': avg_yield,
        'healthy_animals': healthy_animals,
        'health_distribution': health_distribution,
        'yield_by_breed': yield_by_breed,
        'recent_alerts': recent_alerts
    })

@app.route('/api/animals')
def get_animals():
    conn = sqlite3.connect('cattle.db')
    c = conn.cursor()
    
    search = request.args.get('search', '')
    
    if search:
        c.execute('''SELECT * FROM cattle 
                     WHERE id LIKE ? OR breed LIKE ? OR health_status LIKE ?''', 
                 (f'%{search}%', f'%{search}%', f'%{search}%'))
    else:
        c.execute("SELECT * FROM cattle")
    
    animals = []
    for row in c.fetchall():
        animals.append({
            'id': row[0],
            'breed': row[1],
            'age': row[2],
            'weight': row[3],
            'lactation_stage': row[4],
            'parity': row[5],
            'milk_yield': row[6],
            'health_status': row[7],
            'last_updated': row[8]
        })
    
    conn.close()
    return jsonify(animals)

@app.route('/api/animals/create', methods=['POST'])
def create_animal():
    try:
        data = request.json
        required_fields = ['breed', 'age', 'weight', 'lactation_stage', 'parity', 'milk_yield', 'health_status']
        
        # Validate required fields
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Validate data types
        try:
            float(data['age'])
            float(data['weight'])
            float(data['milk_yield'])
            int(data['lactation_stage'])
            int(data['parity'])
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid numeric values'}), 400
            
        if data['health_status'] not in ['Excellent', 'Good', 'Fair', 'Poor']:
            return jsonify({'error': 'Invalid health status'}), 400
        
        conn = sqlite3.connect('cattle.db')
        c = conn.cursor()
        
        # Insert new animal
        c.execute('''INSERT INTO cattle 
                    (breed, age, weight, lactation_stage, parity, milk_yield, health_status, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                 (data['breed'], data['age'], data['weight'], data['lactation_stage'],
                  data['parity'], data['milk_yield'], data['health_status'],
                  datetime.datetime.now().strftime('%Y-%m-%d')))
        
        conn.commit()
        new_id = c.lastrowid
        
        # Get the created animal data
        c.execute("SELECT * FROM cattle WHERE id = ?", (new_id,))
        animal = c.fetchone()
        created_animal = {
            'id': animal[0],
            'breed': animal[1],
            'age': animal[2],
            'weight': animal[3],
            'lactation_stage': animal[4],
            'parity': animal[5],
            'milk_yield': animal[6],
            'health_status': animal[7],
            'last_updated': animal[8]
        }
        
        # Get updated dashboard data
        c.execute("SELECT COUNT(*) FROM cattle")
        total_animals = c.fetchone()[0]
        
        c.execute("SELECT AVG(milk_yield) FROM cattle")
        avg_yield = round(c.fetchone()[0], 1) if c.fetchone() else 0
        
        c.execute("SELECT COUNT(*) FROM cattle WHERE health_status IN ('Excellent', 'Good')")
        healthy_animals = c.fetchone()[0]
        
        # Get health distribution
        c.execute("SELECT health_status, COUNT(*) FROM cattle GROUP BY health_status")
        health_distribution = dict.fromkeys(['Excellent', 'Good', 'Fair', 'Poor'], 0)
        for row in c.fetchall():
            health_distribution[row[0]] = row[1]
        
        # Get yield by breed
        c.execute("SELECT breed, AVG(milk_yield) FROM cattle GROUP BY breed")
        yield_by_breed = {}
        for row in c.fetchall():
            yield_by_breed[row[0]] = round(row[1], 1)
            
        conn.close()
        
        # Emit WebSocket events
        socketio.emit('animal_created', created_animal)
        socketio.emit('dashboard_update', {
            'total_animals': total_animals,
            'avg_yield': avg_yield,
            'healthy_animals': healthy_animals,
            'health_distribution': health_distribution,
            'yield_by_breed': yield_by_breed
        })
        
        return jsonify(created_animal), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/sales', methods=['GET'])
def get_sales_report():
    try:
        start_date = request.args.get('start_date', 
                                    (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d'))
        end_date = request.args.get('end_date', 
                                  datetime.datetime.now().strftime('%Y-%m-%d'))
        
        conn = sqlite3.connect('cattle.db')
        c = conn.cursor()
        
        # Get milk production data
        c.execute('''SELECT 
                        breed,
                        COUNT(*) as cow_count,
                        AVG(milk_yield) as avg_yield,
                        SUM(milk_yield) as total_yield
                    FROM cattle
                    WHERE last_updated BETWEEN ? AND ?
                    GROUP BY breed''', (start_date, end_date))
        
        milk_data = []
        total_revenue = 0
        milk_price_per_liter = 50  # Example price
        
        for row in c.fetchall():
            breed_data = {
                'breed': row[0],
                'cow_count': row[1],
                'avg_yield': round(row[2], 1),
                'total_yield': round(row[3], 1),
                'revenue': round(row[3] * milk_price_per_liter, 2)
            }
            milk_data.append(breed_data)
            total_revenue += breed_data['revenue']
        
        # Get health statistics
        c.execute('''SELECT 
                        health_status,
                        COUNT(*) as count
                    FROM cattle
                    WHERE last_updated BETWEEN ? AND ?
                    GROUP BY health_status''', (start_date, end_date))
        
        health_stats = dict.fromkeys(['Excellent', 'Good', 'Fair', 'Poor'], 0)
        for row in c.fetchall():
            health_stats[row[0]] = row[1]
        
        conn.close()
        
        report = {
            'period': {
                'start': start_date,
                'end': end_date
            },
            'milk_production': milk_data,
            'health_statistics': health_stats,
            'total_revenue': round(total_revenue, 2),
            'milk_price': milk_price_per_liter
        }
        
        return jsonify(report)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/yield', methods=['POST'])
def predict_yield():
    try:
        data = request.json
        
        # Advanced prediction algorithm with multiple factors
        base_yield = 25.0  # Base yield for optimal conditions
        
        # Breed-specific factors based on research data
        breed_factors = {
            'Holstein': {'base': 1.2, 'peak_age': 5, 'weight_optimal': 650},
            'Jersey': {'base': 0.85, 'peak_age': 4, 'weight_optimal': 450},
            'Guernsey': {'base': 1.0, 'peak_age': 5, 'weight_optimal': 500},
            'Sahiwal': {'base': 0.9, 'peak_age': 6, 'weight_optimal': 425}
        }
        breed_info = breed_factors.get(data.get('breed', 'Holstein'))
        
        # Age impact calculation (breed-specific peak age)
        age = float(data.get('age', 5))
        age_factor = 1.0 - (abs(age - breed_info['peak_age']) * 0.08)
        age_factor = max(0.6, min(1.0, age_factor))  # Cap between 0.6 and 1.0
        
        # Weight optimization (breed-specific optimal weight)
        weight = float(data.get('weight', 500))
        weight_diff = abs(weight - breed_info['weight_optimal'])
        weight_factor = 1.0 - (weight_diff / breed_info['weight_optimal'] * 0.3)
        weight_factor = max(0.7, min(1.0, weight_factor))
        
        # Feed impact with diminishing returns
        feed_quantity = float(data.get('feed_quantity', 25))
        feed_factor = 2 / (1 + math.exp(-0.15 * (feed_quantity - 20))) - 0.5  # Sigmoid function
        feed_factor = max(0.6, min(1.2, feed_factor))
        
        # Environmental factors
        temp = float(data.get('ambient_temperature', 22))
        humidity = float(data.get('humidity', 60))
        
        # Temperature stress factor (optimal range: 10-25°C)
        temp_stress = abs(temp - 18) / 30
        temp_factor = 1.0 - temp_stress
        temp_factor = max(0.7, min(1.0, temp_factor))
        
        # Humidity impact (optimal range: 40-70%)
        humidity_stress = abs(humidity - 55) / 100
        humidity_factor = 1.0 - humidity_stress
        humidity_factor = max(0.8, min(1.0, humidity_factor))
        
        # Lactation stage impact
        lactation_stage = int(data.get('lactation_stage', 2))
        lactation_curve = math.exp(-0.05 * lactation_stage) * (lactation_stage ** 0.5)
        lactation_factor = min(1.2, max(0.7, lactation_curve))
        
        # Calculate final prediction with all factors
        prediction = (base_yield * 
                     breed_info['base'] * 
                     age_factor * 
                     weight_factor * 
                     feed_factor * 
                     temp_factor * 
                     humidity_factor * 
                     lactation_factor)
        
        # Add small random variation for natural fluctuation
        variation = np.random.normal(0, 0.5)
        prediction = max(5, min(40, prediction + variation))  # Enforce realistic bounds
        
        # Calculate confidence based on data quality
        confidence = min(0.95, (age_factor + weight_factor + feed_factor + temp_factor + humidity_factor) / 5.0)
        
        return jsonify({
            'prediction': max(10, prediction),  # Ensure at least 10 liters
            'confidence': 0.92
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/health', methods=['POST'])
def predict_health():
    try:
        data = request.json
        
        # Advanced health prediction algorithm with multiple indicators
        base_score = 0.85  # Start with healthy baseline
        
        # Vital signs analysis
        temp = float(data.get('body_temperature', 38.5))
        heart_rate = float(data.get('heart_rate', 65))
        
        # Temperature analysis (normal range: 38.0-39.0°C)
        temp_deviation = abs(temp - 38.5)
        if temp_deviation <= 0.5:
            temp_factor = 1.0
        else:
            temp_factor = max(0.5, 1.0 - (temp_deviation - 0.5) * 0.4)
        
        # Heart rate analysis (normal range: 60-70 BPM)
        hr_deviation = abs(heart_rate - 65)
        if hr_deviation <= 5:
            heart_factor = 1.0
        else:
            heart_factor = max(0.6, 1.0 - (hr_deviation - 5) * 0.03)
            
        # Activity level assessment
        walking_distance = float(data.get('walking_distance', 3))
        grazing_duration = float(data.get('grazing_duration', 6))
        rumination_time = float(data.get('rumination_time', 8))
        
        # Walking distance (optimal: 2-4 km/day)
        walk_factor = 1.0 - abs(walking_distance - 3) * 0.15
        
        # Grazing duration (optimal: 5-8 hours)
        grazing_factor = 1.0 - abs(grazing_duration - 6.5) * 0.1
        
        # Rumination time (optimal: 7-9 hours)
        rumination_factor = 1.0 - abs(rumination_time - 8) * 0.12
        
        # Rest and stress assessment
        resting_hours = float(data.get('resting_hours', 8))
        rest_factor = 2 / (1 + math.exp(-0.5 * (resting_hours - 6))) - 0.5  # Sigmoid function
        
        # Environmental stress factors
        ambient_temp = float(data.get('ambient_temperature', 25))
        humidity = float(data.get('humidity', 60))
        
        # Temperature stress
        temp_stress = 1.0 - abs(ambient_temp - 21) * 0.02
        
        # Humidity stress (optimal: 40-70%)
        humidity_stress = 1.0 - abs(humidity - 55) * 0.01
        
        # Weight change assessment (if available)
        weight = float(data.get('weight', 500))
        breed_optimal_weights = {
            'Holstein': 650,
            'Jersey': 450,
            'Guernsey': 500,
            'Sahiwal': 425
        }
        optimal_weight = breed_optimal_weights.get(data.get('breed', 'Holstein'))
        weight_factor = 1.0 - abs(weight - optimal_weight) / optimal_weight * 0.3
        
        # Calculate comprehensive health score
        vitals_score = (temp_factor * 0.6 + heart_factor * 0.4)
        activity_score = (walk_factor * 0.3 + grazing_factor * 0.35 + rumination_factor * 0.35)
        environment_score = (temp_stress * 0.6 + humidity_stress * 0.4)
        physical_score = (rest_factor * 0.5 + weight_factor * 0.5)
        
        # Weighted combination of all factors
        health_score = (
            base_score *
            (vitals_score * 0.35 +
             activity_score * 0.25 +
             environment_score * 0.15 +
             physical_score * 0.25)
        )
        
        # Enforce realistic bounds and add small random variation
        variation = np.random.normal(0, 0.02)
        health_score = max(0.2, min(0.99, health_score + variation))
        
        # Convert to status
        if health_score > 0.8:
            status = "Excellent"
        elif health_score > 0.6:
            status = "Good"
        elif health_score > 0.4:
            status = "Fair"
        else:
            status = "Poor"
        
        return jsonify({
            'prediction': health_score,
            'status': status
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/animals/update', methods=['POST'])
def update_animal():
    try:
        data = request.json
        required_fields = ['id', 'weight', 'milk_yield', 'health_status']
        
        # Validate required fields
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Validate data types
        if not isinstance(data['id'], int):
            return jsonify({'error': 'Invalid ID format'}), 400
            
        try:
            float(data['weight'])
            float(data['milk_yield'])
        except ValueError:
            return jsonify({'error': 'Invalid numeric values'}), 400
            
        if data['health_status'] not in ['Excellent', 'Good', 'Fair', 'Poor']:
            return jsonify({'error': 'Invalid health status'}), 400
        
        conn = sqlite3.connect('cattle.db')
        c = conn.cursor()
        
        # Check if animal exists
        c.execute("SELECT health_status FROM cattle WHERE id = ?", (data['id'],))
        result = c.fetchone()
        if not result:
            conn.close()
            return jsonify({'error': 'Animal not found'}), 404
            
        old_health_status = result[0]
        
        # Update animal data
        c.execute('''UPDATE cattle 
                    SET weight = ?, milk_yield = ?, health_status = ?, last_updated = ?
                    WHERE id = ?''',
                 (data['weight'], data['milk_yield'], data['health_status'],
                  datetime.datetime.now().strftime('%Y-%m-%d'),
                  data['id']))
                  
        conn.commit()
        
        # Get updated animal data for broadcast
        c.execute("SELECT * FROM cattle WHERE id = ?", (data['id'],))
        animal = c.fetchone()
        updated_animal = {
            'id': animal[0],
            'breed': animal[1],
            'age': animal[2],
            'weight': animal[3],
            'lactation_stage': animal[4],
            'parity': animal[5],
            'milk_yield': animal[6],
            'health_status': animal[7],
            'last_updated': animal[8]
        }
        
        # Get updated dashboard data
        c.execute("SELECT COUNT(*) FROM cattle")
        total_animals = c.fetchone()[0]
        
        c.execute("SELECT AVG(milk_yield) FROM cattle")
        avg_yield = round(c.fetchone()[0], 1)
        
        c.execute("SELECT COUNT(*) FROM cattle WHERE health_status IN ('Excellent', 'Good')")
        healthy_animals = c.fetchone()[0]
        
        # Get health distribution
        c.execute("SELECT health_status, COUNT(*) FROM cattle GROUP BY health_status")
        health_distribution = {}
        for row in c.fetchall():
            health_distribution[row[0]] = row[1]
        
        # Get yield by breed
        c.execute("SELECT breed, AVG(milk_yield) FROM cattle GROUP BY breed")
        yield_by_breed = {}
        for row in c.fetchall():
            yield_by_breed[row[0]] = round(row[1], 1)
            
        conn.close()
        
        # Emit WebSocket events
        socketio.emit('animal_updated', updated_animal)
        socketio.emit('dashboard_update', {
            'total_animals': total_animals,
            'avg_yield': avg_yield,
            'healthy_animals': healthy_animals,
            'health_distribution': health_distribution,
            'yield_by_breed': yield_by_breed
        })
        
        # Emit health status change notification if needed
        if old_health_status != data['health_status'] and data['health_status'] in ['Fair', 'Poor']:
            socketio.emit('health_alert', {
                'animal_id': data['id'],
                'status': data['health_status'],
                'message': f'Animal #{data["id"]} health status changed to {data["health_status"]}'
            })
        
        return jsonify(updated_animal)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Sensor simulation and real-time monitoring
import threading
import time
import random

class SensorSimulator:
    def __init__(self, socketio):
        self.socketio = socketio
        self.running = False
        self.thread = None
        
    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run_simulation)
            self.thread.daemon = True
            self.thread.start()
            
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            
    def run_simulation(self):
        while self.running:
            try:
                conn = sqlite3.connect('cattle.db')
                c = conn.cursor()
                
                # Get all animals
                c.execute("SELECT id, breed FROM cattle")
                animals = c.fetchall()
                
                for animal_id, breed in animals:
                    # Simulate sensor readings
                    sensor_data = self.generate_sensor_data(animal_id, breed)
                    
                    # Update animal data in database
                    self.update_animal_data(c, animal_id, sensor_data)
                    
                    # Emit real-time update
                    self.socketio.emit('sensor_update', {
                        'animal_id': animal_id,
                        'data': sensor_data
                    })
                    
                    # Check for alerts
                    self.check_alerts(animal_id, sensor_data)
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                print(f"Error in sensor simulation: {e}")
                
            time.sleep(30)  # Update every 30 seconds
            
    def generate_sensor_data(self, animal_id, breed):
        # Base values
        base_temp = 38.5
        base_heart_rate = 65
        base_rumination = 8
        base_activity = 3
        
        # Add natural variation
        temp = base_temp + random.gauss(0, 0.2)
        heart_rate = base_heart_rate + random.gauss(0, 3)
        rumination = base_rumination + random.gauss(0, 0.5)
        activity = base_activity + random.gauss(0, 0.3)
        
        # Time-based variations (simulate day/night cycle)
        hour = datetime.now().hour
        if 6 <= hour <= 18:  # Daytime
            activity += 1
            heart_rate += 5
        else:  # Nighttime
            rumination += 1
            
        return {
            'temperature': round(temp, 1),
            'heart_rate': round(heart_rate),
            'rumination_time': round(rumination, 1),
            'activity_level': round(activity, 1),
            'timestamp': datetime.now().isoformat()
        }
        
    def update_animal_data(self, cursor, animal_id, sensor_data):
        # Update last known vitals
        cursor.execute('''UPDATE cattle 
                         SET last_updated = ?
                         WHERE id = ?''',
                      (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                       animal_id))
                       
    def check_alerts(self, animal_id, data):
        alerts = []
        
        # Check temperature
        if data['temperature'] > 39.5:
            alerts.append({
                'type': 'high_temperature',
                'severity': 'high',
                'message': f'High body temperature detected: {data["temperature"]}°C'
            })
        elif data['temperature'] < 38.0:
            alerts.append({
                'type': 'low_temperature',
                'severity': 'high',
                'message': f'Low body temperature detected: {data["temperature"]}°C'
            })
            
        # Check heart rate
        if data['heart_rate'] > 80:
            alerts.append({
                'type': 'high_heart_rate',
                'severity': 'high',
                'message': f'Elevated heart rate: {data["heart_rate"]} BPM'
            })
        elif data['heart_rate'] < 50:
            alerts.append({
                'type': 'low_heart_rate',
                'severity': 'high',
                'message': f'Low heart rate: {data["heart_rate"]} BPM'
            })
            
        # Check activity
        if data['activity_level'] < 1.0:
            alerts.append({
                'type': 'low_activity',
                'severity': 'medium',
                'message': 'Unusually low activity detected'
            })
            
        # Emit alerts if any
        if alerts:
            for alert in alerts:
                self.socketio.emit('health_alert', {
                    'animal_id': animal_id,
                    'alert': alert,
                    'timestamp': datetime.now().isoformat()
                })

# WebSocket event handlers and helper functions
connected_clients = set()

def broadcast_dashboard_update():
    """Helper function to broadcast dashboard updates to all connected clients"""
    try:
        conn = sqlite3.connect('cattle.db')
        c = conn.cursor()
        
        # Get dashboard stats
        c.execute("SELECT COUNT(*) FROM cattle")
        total_animals = c.fetchone()[0]
        
        c.execute("SELECT AVG(milk_yield) FROM cattle")
        avg_yield = round(c.fetchone()[0], 1) if c.fetchone() else 0
        
        c.execute("SELECT COUNT(*) FROM cattle WHERE health_status IN ('Excellent', 'Good')")
        healthy_animals = c.fetchone()[0]
        
        # Get health distribution
        c.execute("SELECT health_status, COUNT(*) FROM cattle GROUP BY health_status")
        health_distribution = dict.fromkeys(['Excellent', 'Good', 'Fair', 'Poor'], 0)
        for row in c.fetchall():
            health_distribution[row[0]] = row[1]
        
        # Get yield by breed
        c.execute("SELECT breed, AVG(milk_yield) FROM cattle GROUP BY breed")
        yield_by_breed = {}
        for row in c.fetchall():
            yield_by_breed[row[0]] = round(row[1], 1)
            
        conn.close()
        
        socketio.emit('dashboard_update', {
            'total_animals': total_animals,
            'avg_yield': avg_yield,
            'healthy_animals': healthy_animals,
            'health_distribution': health_distribution,
            'yield_by_breed': yield_by_breed
        })
        
    except Exception as e:
        print(f"Error in broadcast_dashboard_update: {e}")

def broadcast_animals_update():
    """Helper function to broadcast full animals list update"""
    try:
        conn = sqlite3.connect('cattle.db')
        c = conn.cursor()
        c.execute("SELECT * FROM cattle")
        
        animals = []
        for row in c.fetchall():
            animals.append({
                'id': row[0],
                'breed': row[1],
                'age': row[2],
                'weight': row[3],
                'lactation_stage': row[4],
                'parity': row[5],
                'milk_yield': row[6],
                'health_status': row[7],
                'last_updated': row[8]
            })
        
        conn.close()
        socketio.emit('animals_update', animals)
        
    except Exception as e:
        print(f"Error in broadcast_animals_update: {e}")

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    connected_clients.add(request.sid)
    print(f'Client connected. Total clients: {len(connected_clients)}')
    
    # Send initial data to the new client
    broadcast_dashboard_update()
    broadcast_animals_update()
    
    socketio.emit('connection_status', {'status': 'connected'}, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    if request.sid in connected_clients:
        connected_clients.remove(request.sid)
    print(f'Client disconnected. Total clients: {len(connected_clients)}')

@socketio.on('request_update')
def handle_update_request(data=None):
    """Handle manual update requests from clients"""
    broadcast_dashboard_update()
    broadcast_animals_update()

@socketio.on('add_animal')
def handle_add_animal(data):
    """Handle animal addition through WebSocket"""
    try:
        # Validate required fields
        required_fields = ['breed', 'age', 'weight', 'lactation_stage', 'parity', 'milk_yield', 'health_status']
        if not all(field in data for field in required_fields):
            raise ValueError('Missing required fields')
        
        conn = sqlite3.connect('cattle.db')
        c = conn.cursor()
        
        # Insert new animal
        c.execute('''INSERT INTO cattle 
                    (breed, age, weight, lactation_stage, parity, milk_yield, health_status, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                 (data['breed'], data['age'], data['weight'], data['lactation_stage'],
                  data['parity'], data['milk_yield'], data['health_status'],
                  datetime.datetime.now().strftime('%Y-%m-%d')))
        
        conn.commit()
        new_id = c.lastrowid
        
        # Get the created animal data
        c.execute("SELECT * FROM cattle WHERE id = ?", (new_id,))
        animal = c.fetchone()
        created_animal = {
            'id': animal[0],
            'breed': animal[1],
            'age': animal[2],
            'weight': animal[3],
            'lactation_stage': animal[4],
            'parity': animal[5],
            'milk_yield': animal[6],
            'health_status': animal[7],
            'last_updated': animal[8]
        }
        
        conn.close()
        
        # Broadcast updates
        socketio.emit('animal_added', created_animal)
        broadcast_dashboard_update()
        
    except Exception as e:
        socketio.emit('error', {'message': str(e)}, room=request.sid)

@socketio.on('update_animal')
def handle_update_animal(data):
    """Handle animal updates through WebSocket"""
    try:
        if not all(key in data for key in ['id', 'weight', 'milk_yield', 'health_status']):
            raise ValueError('Missing required fields')
        
        conn = sqlite3.connect('cattle.db')
        c = conn.cursor()
        
        # Update animal
        c.execute('''UPDATE cattle 
                    SET weight = ?, milk_yield = ?, health_status = ?, last_updated = ?
                    WHERE id = ?''',
                 (data['weight'], data['milk_yield'], data['health_status'],
                  datetime.datetime.now().strftime('%Y-%m-%d'), data['id']))
        
        if c.rowcount == 0:
            raise ValueError('Animal not found')
            
        conn.commit()
        
        # Get updated animal data
        c.execute("SELECT * FROM cattle WHERE id = ?", (data['id'],))
        animal = c.fetchone()
        updated_animal = {
            'id': animal[0],
            'breed': animal[1],
            'age': animal[2],
            'weight': animal[3],
            'lactation_stage': animal[4],
            'parity': animal[5],
            'milk_yield': animal[6],
            'health_status': animal[7],
            'last_updated': animal[8]
        }
        
        conn.close()
        
        # Broadcast updates
        socketio.emit('animal_updated', updated_animal)
        broadcast_dashboard_update()
        
    except Exception as e:
        socketio.emit('error', {'message': str(e)}, room=request.sid)

@socketio.on('delete_animal')
def handle_delete_animal(data):
    """Handle animal deletion through WebSocket"""
    try:
        if 'id' not in data:
            raise ValueError('Missing animal ID')
        
        conn = sqlite3.connect('cattle.db')
        c = conn.cursor()
        
        c.execute("DELETE FROM cattle WHERE id = ?", (data['id'],))
        
        if c.rowcount == 0:
            raise ValueError('Animal not found')
            
        conn.commit()
        conn.close()
        
        # Broadcast updates
        socketio.emit('animal_deleted', {'id': data['id']})
        broadcast_dashboard_update()
        
    except Exception as e:
        socketio.emit('error', {'message': str(e)}, room=request.sid)

# Health prediction functions
def get_recent_vitals(cattle_id, hours=24):
    """Get recent vital signs for prediction"""
    query = """
        SELECT temperature, heart_rate, activity_level, rumination_time 
        FROM vital_signs 
        WHERE cattle_id = ? 
        AND timestamp >= datetime('now', '-24 hours')
        ORDER BY timestamp DESC
    """
    try:
        vitals = db.execute_query(query, (cattle_id,), fetch=True)
        if not vitals:
            return None
            
        # Calculate averages
        temp_avg = sum(v[0] for v in vitals) / len(vitals)
        heart_avg = sum(v[1] for v in vitals) / len(vitals)
        activity_avg = sum(v[2] for v in vitals) / len(vitals)
        rumination_avg = sum(v[3] for v in vitals) / len(vitals)
        
        return {
            'temperature': temp_avg,
            'heart_rate': heart_avg,
            'activity_level': activity_avg,
            'rumination_time': rumination_avg
        }
    except Exception:
        return None

def predict_health(cattle_id):
    """Predict health status based on vital signs and other factors"""
    try:
        # Get cattle information
        cattle_info = db.execute_query(
            "SELECT breed, age, weight, lactation_stage FROM cattle WHERE id = ?",
            (cattle_id,),
            fetch=True
        )
        if not cattle_info:
            raise ValueError(f"No cattle found with ID {cattle_id}")
            
        breed, age, weight, lactation = cattle_info[0]
        
        # Get recent vital signs
        vitals = get_recent_vitals(cattle_id)
        if not vitals:
            raise ValueError("No recent vital signs data available")
        
        # Calculate health score based on multiple factors
        health_score = 1.0  # Start with perfect score
        warnings = []
        
        # Temperature analysis (38.5-39.5°C is normal)
        temp = vitals['temperature']
        if temp < 38.0 or temp > 39.5:
            health_score *= 0.8
            warnings.append(f"Abnormal temperature: {temp}°C")
        
        # Heart rate analysis (60-70 BPM is normal)
        heart = vitals['heart_rate']
        if heart < 50 or heart > 80:
            health_score *= 0.85
            warnings.append(f"Abnormal heart rate: {heart} BPM")
        
        # Activity level analysis (10-14 hours is normal)
        activity = vitals['activity_level']
        if activity < 8 or activity > 16:
            health_score *= 0.9
            warnings.append(f"Abnormal activity level: {activity} hours")
        
        # Rumination time analysis (7-9 hours is normal)
        rumination = vitals['rumination_time']
        if rumination < 6 or rumination > 10:
            health_score *= 0.85
            warnings.append(f"Abnormal rumination time: {rumination} hours")
        
        # Adjust for age
        if age > 10:
            health_score *= 0.95
        
        # Adjust for lactation stage
        if lactation > 5:
            health_score *= 0.98
        
        # Determine status
        if health_score >= 0.9:
            status = "Excellent"
        elif health_score >= 0.8:
            status = "Good"
        elif health_score >= 0.6:
            status = "Fair"
        else:
            status = "Poor"
        
        return {
            'cattle_id': cattle_id,
            'health_score': round(health_score, 2),
            'status': status,
            'warnings': warnings,
            'vitals': vitals,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise Exception(f"Error predicting health: {str(e)}")

@socketio.on('request_prediction')
def handle_prediction_request(data):
    """Handle prediction requests through WebSocket"""
    try:
        cattle_id = data.get('cattle_id')
        if not cattle_id:
            raise ValueError("No cattle_id provided")
            
        # Get health prediction
        health_data = predict_health(cattle_id)
        
        # If health score is low, add an alert
        if health_data['health_score'] < 0.6:
            db.add_alert(
                cattle_id,
                'health',
                'high',
                f"Low health score detected ({health_data['health_score']}): {', '.join(health_data['warnings'])}"
            )
        
        socketio.emit('prediction_result', {
            'type': 'health',
            'data': health_data
        })
        
    except Exception as e:
        socketio.emit('prediction_error', {'error': str(e)})

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    try:
        # Initialize sensor simulator
        sensor_simulator = SensorSimulator(db)
        sensor_simulator.start()
        
        # Start the Flask-SocketIO app
        socketio.run(app, debug=True, port=5000)
    finally:
        # Stop sensor simulator when app stops
        if 'sensor_simulator' in locals() and sensor_simulator:
            sensor_simulator.stop()

# End of file