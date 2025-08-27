SQL-DB: MySQL/Oracle/Postgre/MS-SQL server    RDBMs (Databases) - OLTP Type
Data-Warehouse: GCP-BigQuery/Azure-Synapse/AWS-RedShift/SnowFlake  - OLAP Type    (For Data Analytics of Past huge data)
Database (Structured data), Data-Lake[Storage for Any(Unstructured/Semi/Strutured) huge data Ex: S3, GCS, Blob, TB/PB/EB], DataWarehouse(Structured data huge data GB/TB)
# SQL Commands: [DDL(definition), DML(Manipulation), DRL(Retrieval), TCL(Transaction Control), DCL(Data Control)]
DDL(Create, Alter, Drop), DML(Insert, Update, Delete), DRL(Select), TCL(Commit, Rollback), DCL(Grant, Revoke)
AggregateFunctions:
Constraints: Not Null, Unique, Primary key, Foreign Key, Check, Default

Basics:
create database if not exists nn-db; --> use nn-db; --> drop database nn-db; --> show databases;
create table nn_emp(id int, name varchar(20), email varchar(30), age int);
select * from nn_emp;
create table emp1(id int, email varchar(28) not null, phone int unique)
create table emp2(id int, Age int check(Age>18), Country varchar(28) default 'INDIA')
insert into emp2 values(1,17,'India')
# primary key (Not be Null, Not be duplicate, only one column should have primary key) & unique key (Not be duplicate, can be null, Many columns having unique keys)
create table emp3(id int primary key, email varchar(28) unique)
create table department(Dept_id int primary key, Dept_Name varchar(28))
create table student(id int primary key, Name varchar(28), Dept_id int, foreign key (Dept_id) references department(Dept_id))
update student set address='Dehradun' where id=21;
desc student;
alter table student modify column name varchar(60);    # it will change varchar length of name column from 30 to 60
alter table student drop column marks;    # here marks column will be deleted
delete from student where name='Murty'; 
alter table nn_emp add(gender varchar(10))
alter table nn_emp add(city varchar(30), Dept varchar(30), DOJ date);
insert into nn_emp(id, name, email, age, gender, city, Dept, DOJ) values(1,'Sachin','sachin123@gmail.com',27,'M',Mumbai,'Sports',1996-01-06);
drop table nn_emp;
insert into nn_emp values(2,'Ganguly','Ganguly623@gmail.com',32,'M',Kolkata,'Sports',1991-09-25);
insert into nn_emp 
values(3,'KapilDev','kapildev007@gmail.com',36,'M',Mumbai,'Sports',1987-09-25),
(4,'Ajaruddin','azar@gmail.com',29,'M',Mumbai,'Sports',1994-06-06),
(5,'Dravid','dravid@gmail.com',27,'M',Mumbai,'Sports',1995-08-21);
truncate table nn_emp;
select id, name, email, city from nn_emp where city='Mumbai';               # where with multiple columns
select Dept as Department, DOJ as DateOfJoining from nn_emp where gender='M';       # alias (as)
select * from orders;
select * from orders where customerNumber=121;
select * from orders where customerNumber > 121;
select * from orders where customerNumber >= 121;
select * from orders where customerNumber <= 121;
select * from orders where customerNumber != 121; (or) select * from orders where customerNumber <> 121;
select * from products where buyPrice BETWEEN 80 and 100;
select * from products where buyPrice NOT BETWEEN 80 and 100;
select * from products where buyPrice IN 80, 85, 90, 95, 100;
select * from customers where country = 'USA' and state = 'CA';
select * from customers where country = 'USA' and state = 'CA' and creditlimit > 100000;
select * from customers where country = 'USA' or country = 'France' and creditlimit > 100000;
select * from customers where state is NULL order by customerName asc;
select * from customers where state is not NULL order by customerName asc;
select * from customers where country in ('USA', 'France', 'UK');
select count(buyPrice) from products;                                   # Aggregate Functions [ count, Min, Max, Avg]
select min(buyPrice) from products;
select max(buyPrice) from products; (or) select max(buyPrice) as Max_Price from products;
select max(buyPrice) from products where productLine = 'Motorcycles';
select sum(buyPrice) as Total_BuyAmount from products;
select avg(buyPrice) as Avg_Price from products;
select avg(buyPrice) as Avg_Price from products where productLine = 'Ships';
select * from products where productVendor like 'r%';
select * from products where productVendor like '%in%';
select * from products where productVendor like 'c%s';
create table student1(student_id int Not null auto_increment, Name varchar(30), Marks int, primary key(student_id));
insert into student1 values('Guru', 85)             # here No need to give student_id for each record...it will take automatic 1,2,3....
select * from products group by productLine;        # here we can just group the productLine column products wise
select count(*) from products group by productLine; # here in addition to group and can give the count of each product in that column/Table
select productLine, count(productLine) as count from products group by productLine;
select productLine, count(productLine) as count from products group by productLine order by count desc; # here we can get descending order of count results
select productLine, count(productLine) as count from products where buyPrice>90 group by productLine order by count desc;   # here giving counts of products (descending order) whose buyprice>90 only
select * from customers order by customerName;      # It sorting customers data in ascending order in customerName records
select contactFirstName, contactLastName from customers order by contactFirstName asc, contactLastName desc;
select qtyOrdered, priceEach, qtyOrdered*priceEach as Total from orderdetails order by Total desc;
select distinct country from customers;
select productLine, count(productLine) as count from products group by productLine order by count desc; # order of Execution: from --> where --> group by --> select --> order by
# aggregation cannot be used in where clause, but can use in Having clause. 'Where' comes before 'group by', Having comes after 'group by'
# we can use where clause without group by also, but Having must with group by  [ where & having both for Filtering only]
select productLine, count(productLine) as count from products where buyPrice>50 group by productLine having count>5 order by count desc;
# (order of Execution: from --> where --> group by --> having --> select --> order by)
select country, creditLimit, count(country) as count from customers where creditLimit > 100000 group by country having count>2;
select orderNumber, sum(qtyOrdered) as Total_Qty, sum(qtyOrdered*priceEach) as Total_Amt from orderdetails group by orderNumber;
select orderNumber, sum(qtyOrdered) as Total_Qty, sum(qtyOrdered*priceEach) as Total_Amt from orderdetails group by orderNumber having Total_Amt > 5000;

Advance:
