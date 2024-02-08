/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_52b.c
Label Definition File: CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic.label.xml
Template File: sources-sinks-52b.tmpl.c
*/
/*
 * @description
 * CWE: 135 Incorrect Calculation of Multi-Byte String Length
 * BadSource:  Void pointer to a wchar_t array
 * GoodSource: Void pointer to a char array
 * Sinks:
 *    GoodSink: Allocate memory using wcslen() and copy data
 *    BadSink : Allocate memory using strlen() and copy data
 * Flow Variant: 52 Data flow: data passed as an argument from one function to another to another in three different source files
 *
 * */

#include "std_testcase.h"

# include <wchar.h>

#define WIDE_STRING L"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
#define CHAR_STRING "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"

#ifndef OMITBAD

/* bad function declaration */
void CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_52c_bad_sink(void * data);

void CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_52b_bad_sink(void * data)
{
    CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_52c_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_52c_goodG2B_sink(void * data);

void CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_52b_goodG2B_sink(void * data)
{
    CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_52c_goodG2B_sink(data);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_52c_goodB2G_sink(void * data);

void CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_52b_goodB2G_sink(void * data)
{
    CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_52c_goodB2G_sink(data);
}

#endif /* OMITGOOD */
