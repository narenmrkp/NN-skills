sudo apt update
sudo apt install postgresql postgresql-contrib -y
sudo -i -u postgres
psql --> \q --> exit
sudo -u postgres psql
ALTER USER postgres WITH PASSWORD 'naren@123';
CREATE DATABASE my_database;
\c my_database;
CREATE TABLE users (
   id SERIAL PRIMARY KEY,
   name VARCHAR(100),
   email VARCHAR(100)
);

-- Insert 50 dummy data entries into the users table
INSERT INTO users (name, email) VALUES
('Alice Johnson', 'alice.johnson@example.com'),
('Bob Smith', 'bob.smith@example.com'),
('Charlie Brown', 'charlie.brown@example.com'),
('Diana Prince', 'diana.prince@example.com'),
('Ethan Hunt', 'ethan.hunt@example.com'),
('Fiona Gallagher', 'fiona.gallagher@example.com'),
('George Washington', 'george.washington@example.com'),
('Hannah Montana', 'hannah.montana@example.com'),
('Ian Malcolm', 'ian.malcolm@example.com'),
('Julia Roberts', 'julia.roberts@example.com'),
('Kevin Bacon', 'kevin.bacon@example.com'),
('Laura Croft', 'laura.croft@example.com'),
('Michael Scott', 'michael.scott@example.com'),
('Nina Simone', 'nina.simone@example.com'),
('Oscar Wilde', 'oscar.wilde@example.com'),
('Paula Abdul', 'paula.abdul@example.com'),
('Quentin Tarantino', 'quentin.tarantino@example.com'),
('Rachel Green', 'rachel.green@example.com'),
('Steve Jobs', 'steve.jobs@example.com'),
('Tina Fey', 'tina.fey@example.com'),
('Uma Thurman', 'uma.thurman@example.com'),
('Victor Hugo', 'victor.hugo@example.com'),
('Wanda Maximoff', 'wanda.maximoff@example.com'),
('Xena Warrior', 'xena.warrior@example.com'),
('Yara Shahidi', 'yara.shahidi@example.com'),
('Zach Galifianakis', 'zach.galifianakis@example.com'),
('Alice Cooper', 'alice.cooper@example.com'),
('Bob Marley', 'bob.marley@example.com'),
('Cathy Freeman', 'cathy.freeman@example.com'),
('David Beckham', 'david.beckham@example.com'),
('Eva Mendes', 'eva.mendes@example.com'),
('Frank Sinatra', 'frank.sinatra@example.com'),
('Gina Rodriguez', 'gina.rodriguez@example.com'),
('Henry Cavill', 'henry.cavill@example.com'),
('Isla Fisher', 'isla.fisher@example.com'),
('Jack Sparrow', 'jack.sparrow@example.com'),
('Kylie Jenner', 'kylie.jenner@example.com'),
('Leonardo DiCaprio', 'leonardo.dicaprio@example.com'),
('Megan Fox', 'megan.fox@example.com'),
('Nicolas Cage', 'nicolas.cage@example.com'),
('Olivia Wilde', 'olivia.wilde@example.com'),
('Pablo Picasso', 'pablo.picasso@example.com'),
('Queen Latifah', 'queen.latifah@example.com'),
('Ryan Gosling', 'ryan.gosling@example.com'),
('Selena Gomez', 'selena.gomez@example.com'),
('Tom Hanks', 'tom.hanks@example.com'),
('Uma Thurman', 'uma.thurman@example.com'),
('Vin Diesel', 'vin.diesel@example.com'),
('Will Smith', 'will.smith@example.com'),
('Xander Cage', 'xander.cage@example.com'),
('Yasmine Bleeth', 'yasmine.bleeth@example.com'),
('Zoe Saldana', 'zoe.saldana@example.com');

SELECT * FROM users;
cd /etc/postgresql/16/main/
sudo vi pg_hba.conf --> Update postgres with md5 instead of peer insdie this pg_hba.conf file Esc :wq! --> cd ~
sudo systemctl status postgresql
sudo systemctl reload postgresql

sudo apt install unzip -y
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

aws configure --> give keys, region
vi .env
# .env
PG_USER="postgres"
PG_PASSWORD="naren@123"
PG_DATABASE="my_database"    	# Database to back up
PG_TARGET_DATABASE="new_database"   # Database to restore to
S3_BUCKET="nnbkt2025"
S3_PATH="nnbackup"
Esc wq! (save)

vi backup.sh
#!/bin/bash
# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)
# Backup Script
BACKUP_FILE="backup_$(date +"%Y%m%d_%H%M%S").sql"
# Perform database backup
PGPASSWORD="$PG_PASSWORD" pg_dump -U "$PG_USER" -d "$PG_DATABASE" > "$BACKUP_FILE"
# Compress the backup into a tar.gz file
tar -czvf "$BACKUP_FILE.tar.gz" "$BACKUP_FILE"
# Upload backup to Amazon S3
aws s3 cp "$BACKUP_FILE.tar.gz" "s3://$S3_BUCKET/$S3_PATH/$BACKUP_FILE.tar.gz"
# Check if the upload to S3 was successful
if [ $? -eq 0 ]; then
	echo "Backup uploaded to S3 successfully. Removing local backup files."
	# Remove the original SQL backup file and compressed file
	rm "$BACKUP_FILE"
	rm "$BACKUP_FILE.tar.gz"
else
	echo "Failed to upload backup to S3. Keeping local backup files."
fi

sudo chmod +x backup.sh
./backup.sh

vi restore.sh
#!/bin/bash
# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)
# Get the latest backup file from S3
LATEST_BACKUP_FILE=$(aws s3 ls s3://$S3_BUCKET/$S3_PATH/ | sort | tail -n 1 | awk '{print $4}')
if [ -z "$LATEST_BACKUP_FILE" ]; then
	echo "No backup files found in S3."
	exit 1
fi
# Download the latest backup file from S3
aws s3 cp s3://$S3_BUCKET/$S3_PATH/$LATEST_BACKUP_FILE $LATEST_BACKUP_FILE
# Unzip the backup file
tar -xzvf $LATEST_BACKUP_FILE
# Extract the SQL file name from the tar.gz file
SQL_FILE=$(basename "$LATEST_BACKUP_FILE" .tar.gz)
# Drop the target database if it exists
PGPASSWORD="$PG_PASSWORD" psql -U $PG_USER -c "DROP DATABASE IF EXISTS $PG_TARGET_DATABASE;"
# Create the target database
PGPASSWORD="$PG_PASSWORD" psql -U $PG_USER -d postgres -c "CREATE DATABASE $PG_TARGET_DATABASE;"
# Restore the database backup to the target database
PGPASSWORD="$PG_PASSWORD" psql -U $PG_USER -d $PG_TARGET_DATABASE < "$SQL_FILE"
# Verify the restoration
psql -U $PG_USER -d $PG_TARGET_DATABASE -c "SELECT COUNT(*) FROM users;"
# Remove the backup files if the verification is successful
if [ $? -eq 0 ]; then
	echo "Restoration successful. Removing local backup files."
	rm "$SQL_FILE"
	rm "$LATEST_BACKUP_FILE"
else
	echo "Restoration failed. Keeping local backup files."
fi

sudo chmod +x restore.sh
./restore.sh

sudo -u postgres psql
\c new_database;
SELECT * FROM users;
