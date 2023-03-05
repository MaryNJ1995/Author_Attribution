
From python:latest
WORKDIR .
COPY requirements.txt ./home/
COPY src ./home/src/
COPY data ./home/data/
COPY models ./home/models/
RUN /usr/local/bin/python -m pip install --default-timeout=1200 -i http://pypi.partdp.ir/root/pypi/+simple/ --trusted-host pypi.partdp.ir --upgrade pip
RUN pip install --default-timeout=1200 -i http://pypi.partdp.ir/root/pypi/+simple/ --trusted-host pypi.partdp.ir -r ./home/requirements.txt
#RUN apt-get update
#RUN apt-get install nano
CMD [ "python", "./home/src/app.py" ]





