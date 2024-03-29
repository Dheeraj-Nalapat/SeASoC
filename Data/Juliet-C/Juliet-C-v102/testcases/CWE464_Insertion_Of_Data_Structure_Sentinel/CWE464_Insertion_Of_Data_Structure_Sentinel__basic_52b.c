/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE464_Insertion_Of_Data_Structure_Sentinel__basic_52b.c
Label Definition File: CWE464_Insertion_Of_Data_Structure_Sentinel__basic.label.xml
Template File: sources-sink-52b.tmpl.c
*/
/*
 * @description
 * CWE: 464 Insertion of Data Structure Sentinel
 * BadSource:  Read in data from the console and convert to an int
 * GoodSource: Set data to a fixed char
 * Sink:
 *    BadSink : Place data into and print an array
 * Flow Variant: 52 Data flow: data passed as an argument from one function to another to another in three different source files
 *
 * */

#include "std_testcase.h"

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

/* bad function declaration */
void CWE464_Insertion_Of_Data_Structure_Sentinel__basic_52c_bad_sink(char data);

void CWE464_Insertion_Of_Data_Structure_Sentinel__basic_52b_bad_sink(char data)
{
    CWE464_Insertion_Of_Data_Structure_Sentinel__basic_52c_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declaration */
void CWE464_Insertion_Of_Data_Structure_Sentinel__basic_52c_goodG2B_sink(char data);

/* goodG2B uses the GoodSource with the BadSink */
void CWE464_Insertion_Of_Data_Structure_Sentinel__basic_52b_goodG2B_sink(char data)
{
    CWE464_Insertion_Of_Data_Structure_Sentinel__basic_52c_goodG2B_sink(data);
}

#endif /* OMITGOOD */
