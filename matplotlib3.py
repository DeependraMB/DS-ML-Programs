import matplotlib.pyplot as mplb

categories = ["Deependra", "Sreehari","Subin","pranav"];
values = [1,2,3,4];

mplb.bar(categories,values);

mplb.title("Bar Chart Example")
mplb.legend(["Values"])

mplb.xlabel(categories);
mplb.bar(categories, values)
mplb.xticks(rotation=45)

mplb.ylabel(values);

mplb.show();