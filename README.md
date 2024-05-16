Note: Our work is not targeted at any organization or country, and there is no conflict of interest. However, due to the special nature of defense topics, in order to avoid improper use of the code and possible ethical concerns, we have decided not to open the environment code in public, but only to open the code of the algorithm. If you have any questions, please leave a message in the question area.

The code can be used for multi-ship dynamic weapon target assignment. Familiarity with the simulation platform and related AI interfaces is required to utilize this code effectively. Additionally, the following modifications are recommended:

1、Modify the platform path within the code.

2、Switch the platform mode to AI training mode.

3、Adjust the total threshold in message outputs to 1000, displaying only combat unit AI, weapon terminal calculations, weapon logic, and weapon firing.

4、The platform generates numerous records after each round, but only message outputs are necessary. It's advisable to disable other records in the configuration file.

5、Increase the evasion of the Red side's fighter aircraft to 500 since weapons may automatically target fighters, making it challenging to obtain the index of projectiles (Modify this in the database).

6、Regarding weapon range, modifications can be made in the database as needed.
