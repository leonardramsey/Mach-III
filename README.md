# Mach-III

To run web application:

1. Clone repo
2. Install dependencies via the following command:
	
	```shell
	pip install -r requirements.txt
	```
3. If needed, update the database schema:
	
	```shell
	python mach3/manage.py makemigrations
	python mach3/manage.py migrate
	```
4. If needed, upload initial database records:
	
	```shell
	python mach3/manage.py loaddata learning/fixtures/initial_data.json
	```
5. Start a local server via the following command:
	
	```shell
	python mach3/manage.py runserver
	```
6. Access the web application via the following domain:
	
	```localhost:8000/<path>```
		
