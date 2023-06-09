from csv import writer
import time

import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from nordvpn_switcher import initialize_VPN, rotate_VPN


def main():
    initialize_VPN(stored_settings=1)

    with open("D:/Projeler/Sahibinden/websites.txt", "r") as file:
        websites = file.read().split("\n")

    for hata in range(3):
        try:
            with open("D:/Projeler/Sahibinden/websites.txt", "r") as file:
                websites = file.read().split("\n")

            for index, website in enumerate(websites):
                if index % 10 == 0:
                    rotate_VPN()
                    driver = uc.Chrome()
                    driver.get(website)
                    time.sleep(3)
                    driver.refresh()
                else:
                    driver.get(website)
                try:
                    soup = BeautifulSoup(driver.page_source, "html.parser")
                    links = soup.find_all(
                        "td", {"class": "searchResultsLargeThumbnail"}
                    )
                    with open("D:\\Projeler\\Sahibinden\\links.txt", "a") as file:
                        for link in links:
                            site = link.find("a").get("href")
                            content = site + "\n"
                            file.write(content)
                except:
                    pass
                time.sleep(2)
                if index % 10 == 9:
                    driver.close()

                try:
                    with open("D:\\Projeler\\Sahibinden\\websites.txt",
                              "r") as fr:
                        lines = fr.readlines()

                    with open("D:\\Projeler\\Sahibinden\\websites.txt",
                              "w") as fw:
                        for line in lines:
                            if line.strip("\n") != website:
                                fw.write(line)
                    print(
                        "Deleted "
                        + website
                        + " ("
                        + str(index + 1)
                        + "/"
                        + str(len(websites) + 1)
                        + ")"
                    )
                except:
                    print("Oops! Bir hata meydana geldi")

        except Exception as e:
            print(e)

    while True:
        try:
            with open("D:/Projeler/Sahibinden/links.txt", "r") as file:
                links = file.read().split("\n")

            for index, link in enumerate(links):
                house_link = "https://www.sahibinden.com" + link
                if index % 10 == 0:
                    rotate_VPN()
                    options = uc.ChromeOptions()
                    options.headless = True
                    driver = uc.Chrome(options=options)
                    driver.get(house_link)
                    time.sleep(3)
                    driver.refresh()
                else:
                    driver.get(house_link)
                try:
                    soup = BeautifulSoup(driver.page_source, "html.parser")
                    try:
                        bot_check = soup.find("a",
                                              {"class": "error-page-logo"})
                        if bot_check:
                            driver.close()
                            rotate_VPN()
                            options2 = uc.ChromeOptions()
                            options2.headless = True
                            driver = uc.Chrome(options=options2)
                            driver.get(house_link)
                            time.sleep(3)
                            driver.refresh()
                            soup = BeautifulSoup(driver.page_source,
                                                 "html.parser")
                        else:
                            pass
                    except:
                        pass

                    price = (
                        soup.find("div", {"class": "classifiedInfo"})
                        .find("h3")
                        .get_text()
                        .split("\n")[1]
                        .replace(" ", "")
                        .replace(".", "")
                        .replace("TL", "")
                    )
                    ilce = (
                        soup.find("div", {"class": "classifiedInfo"})
                        .find("h2")
                        .find_all("a")[1]
                        .get_text()
                        .replace(" ", "")
                        .replace("\n", "")
                    )
                    mahalle = (
                        soup.find("div", {"class": "classifiedInfo"})
                        .find("h2")
                        .find_all("a")[2]
                        .get_text()
                        .replace("  ", "")
                        .replace("\n", "")
                    )
                    brut = (
                        soup.find("ul", {"class": "classifiedInfoList"})
                        .find_all("span")[3]
                        .get_text()
                        .replace(" ", "")
                        .replace("\t", "")
                        .replace("\n", "")
                    )
                    net = (
                        soup.find("ul", {"class": "classifiedInfoList"})
                        .find_all("span")[4]
                        .get_text()
                        .replace(" ", "")
                        .replace("\t", "")
                        .replace("\n", "")
                    )
                    oda_sayisi = (
                        soup.find("ul", {"class": "classifiedInfoList"})
                        .find_all("span")[5]
                        .get_text()
                        .replace(" ", "")
                        .replace("\t", "")
                        .replace("\n", "")
                    )
                    bina_yasi = (
                        soup.find("ul", {"class": "classifiedInfoList"})
                        .find_all("span")[6]
                        .get_text()
                        .replace(" ", "")
                        .replace("\t", "")
                        .replace("\n", "")
                    )
                    bulundugu_kat = (
                        soup.find("ul", {"class": "classifiedInfoList"})
                        .find_all("span")[7]
                        .get_text()
                        .replace(" ", "")
                        .replace("\t", "")
                        .replace("\n", "")
                    )
                    kat_sayisi = (
                        soup.find("ul", {"class": "classifiedInfoList"})
                        .find_all("span")[8]
                        .get_text()
                        .replace(" ", "")
                        .replace("\t", "")
                        .replace("\n", "")
                    )
                    isitma = (
                        soup.find("ul", {"class": "classifiedInfoList"})
                        .find_all("span")[9]
                        .get_text()
                        .replace("\t", "")
                        .replace("  ", "")
                        .replace("\n", "")
                    )
                    banyo = (
                        soup.find("ul", {"class": "classifiedInfoList"})
                        .find_all("span")[10]
                        .get_text()
                        .replace("\t", "")
                        .replace("  ", "")
                        .replace("\n", "")
                    )
                    balkon = (
                        soup.find("ul", {"class": "classifiedInfoList"})
                        .find_all("span")[11]
                        .get_text()
                        .replace("\t", "")
                        .replace("  ", "")
                        .replace("\n", "")
                    )
                    esyali = (
                        soup.find("ul", {"class": "classifiedInfoList"})
                        .find_all("span")[12]
                        .get_text()
                        .replace("\t", "")
                        .replace("  ", "")
                        .replace("\n", "")
                    )
                    kullanim_durumu = (
                        soup.find("ul", {"class": "classifiedInfoList"})
                        .find_all("span")[13]
                        .get_text()
                        .replace("\t", "")
                        .replace("  ", "")
                        .replace("\n", "")
                    )
                    site = (
                        soup.find("ul", {"class": "classifiedInfoList"})
                        .find_all("span")[14]
                        .get_text()
                        .replace("\t", "")
                        .replace("  ", "")
                        .replace("\n", "")
                    )
                    aidat = (
                        soup.find("ul", {"class": "classifiedInfoList"})
                        .find_all("span")[16]
                        .get_text()
                        .replace("\t", "")
                        .replace("  ", "")
                        .replace("\n", "")
                    )
                    kredi = (
                        soup.find("ul", {"class": "classifiedInfoList"})
                        .find_all("span")[17]
                        .get_text()
                        .replace("\t", "")
                        .replace("  ", "")
                        .replace("\n", "")
                    )
                    tapu = (
                        soup.find("ul", {"class": "classifiedInfoList"})
                        .find_all("span")[18]
                        .get_text()
                        .replace("\t", "")
                        .replace("  ", "")
                        .replace("\n", "")
                    )
                    ev = [
                        price,
                        ilce,
                        mahalle,
                        brut,
                        net,
                        oda_sayisi,
                        bina_yasi,
                        bulundugu_kat,
                        kat_sayisi,
                        isitma,
                        banyo,
                        balkon,
                        esyali,
                        kullanim_durumu,
                        site,
                        aidat,
                        kredi,
                        tapu,
                    ]

                    with open(
                        "houses.csv", "a", newline="", encoding="utf-8"
                    ) as f_object:
                        writer_object = writer(f_object, delimiter=",")
                        writer_object.writerow(ev)
                        f_object.close()

                except:
                    pass
                time.sleep(2)
                if index % 10 == 9:
                    driver.close()

                try:
                    with open("D:\\Projeler\\Sahibinden\\links.txt",
                              "r") as fr:
                        lines = fr.readlines()

                        with open("D:\\Projeler\\Sahibinden\\links.txt",
                                  "w") as fw:
                            for line in lines:
                                if line.strip("\n") != link:
                                    fw.write(line)
                    print(
                        "Deleted "
                        + link
                        + " ("
                        + str(index + 1)
                        + "/"
                        + str(len(links) + 1)
                        + ")"
                    )
                except:
                    print("Oops! Bir hata meydana geldi")

        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
