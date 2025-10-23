# Ansible setup
# Master




ansible all -m ping --> yes

# Slave



# Postgre Install on Slave
sudo apt install postgresql postgresql-contrib -y
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
sudo vi pg_hba.conf    Update postgres with md5 instead of peer
sudo systemctl status postgresql
sudo systemctl reload postgresql

sudo apt install unzip -y
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# on Master:
vi vars.yml
postgres_user: postgres
postgres_db: my_database
postgres_password: naren@123
s3_bucket: nnbkt2025
s3_prefix: nnbackup
aws_access_key: xxxxxxxxxxxx
aws_secret_key: xxxxxxxxxxxxxx
aws_region: ap-south-1
Esc :wq! (save)

ansible-vault encrypt vars.yml --> give any password for vault
vi pg_backup.yml    # This is ansible playbook, code not provided by mr cloudbook, so we need to take from chatgpt
ansible-playbook pg_backup.yml --ask-vault-pass --> give vault password, Next playbook will run to take backup of postgreSQL DB data into s3 bkt




