# -*- coding: utf-8 -*-
import scrapy
import re

baseurl = 'http://www.chefkoch.de'

class ChefkochSpider(scrapy.Spider):
    name = 'ChefkochSpider'
    allowed_domains = ['www.chefkoch.de']
    start_urls = [ baseurl + '/rs/s0/Rezepte.html' ]
    custom_settings = {
            'FEED_FORMAT' : 'jsonLines',
            }
    

    def parse(self, response):
        text = response.body.decode(response.encoding)

        for link in re.finditer(r'href="(/rezepte/\d{5,}.*?.html)"', text):
            yield scrapy.Request(str(baseurl + link.group(1)), callback=self.parse_recipe)

        for link in response.css('a.qa-pagination-next'):
            yield scrapy.Request(baseurl + link.xpath('@href').extract_first(), callback=self.parse)

            
    def parse_recipe(self, response):
        result = dict()
        result['url'] = response.url
        result['title'] = response.css('h1.page-title::text').extract_first()

        result['ingredients'] = list()

        for ingredient in response.css('table.incredients tr'):
            col = ingredient.xpath('(.//td)[2]')
            links = col.xpath('.//a//text()')
            if len(links) == 0:
                result['ingredients'].append(col.xpath('string(.)').extract_first().strip())
            else:
                result['ingredients'].append(links[0].extract().strip())
                
            
        yield result




