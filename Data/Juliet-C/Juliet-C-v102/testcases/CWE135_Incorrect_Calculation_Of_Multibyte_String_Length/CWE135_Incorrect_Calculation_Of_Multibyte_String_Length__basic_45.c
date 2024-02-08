/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_45.c
Label Definition File: CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic.label.xml
Template File: sources-sinks-45.tmpl.c
*/
/*
 * @description
 * CWE: 135 Incorrect Calculation of Multi-Byte String Length
 * BadSource:  Void pointer to a wchar_t array
 * GoodSource: Void pointer to a char array
 * Sinks:
 *    GoodSink: Allocate memory using wcslen() and copy data
 *    BadSink : Allocate memory using strlen() and copy data
 * Flow Variant: 45 Data flow: data passed as a static global variable from one function to another in the same source file
 *
 * */

#include "std_testcase.h"

# include <wchar.h>

#define WIDE_STRING L"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
#define CHAR_STRING "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"

static void * CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_45_bad_data;
static void * CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_45_goodG2B_data;
static void * CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_45_goodB2G_data;

#ifndef OMITBAD

static void bad_sink()
{
    void * data = CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_45_bad_data;
    {
        /* POTENTIAL FLAW - treating pointer as a char* when it may point to a wide string */
        size_t data_len = strlen((char *)data);
        void * data_dest = (void *)calloc(data_len+1, 1);
        memcpy(data_dest, data, (data_len+1));
        printLine((char *)data_dest);
        free(data_dest);
    }
}

void CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_45_bad()
{
    void * data;
    data = NULL;
    /* POTENTIAL FLAW: Set data to point to a wide string */
    data = WIDE_STRING;
    CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_45_bad_data = data;
    bad_sink();
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B_sink()
{
    void * data = CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_45_goodG2B_data;
    {
        /* POTENTIAL FLAW - treating pointer as a char* when it may point to a wide string */
        size_t data_len = strlen((char *)data);
        void * data_dest = (void *)calloc(data_len+1, 1);
        memcpy(data_dest, data, (data_len+1));
        printLine((char *)data_dest);
        free(data_dest);
    }
}

static void goodG2B()
{
    void * data;
    data = NULL;
    /* FIX: Set data to point to a char string */
    data = CHAR_STRING;
    CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_45_goodG2B_data = data;
    goodG2B_sink();
}

/* goodB2G() uses the BadSource with the GoodSink */
static void goodB2G_sink()
{
    void * data = CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_45_goodB2G_data;
    {
        /* FIX - treating pointer like a wchar_t*  */
        size_t data_len = wcslen((wchar_t *)data);
        void * data_dest = (void *)calloc(data_len+1, sizeof(wchar_t));
        memcpy(data_dest, data, (data_len+1)*sizeof(wchar_t));
        printWLine((wchar_t *)data_dest);
        free(data_dest);
    }
}

static void goodB2G()
{
    void * data;
    data = NULL;
    /* POTENTIAL FLAW: Set data to point to a wide string */
    data = WIDE_STRING;
    CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_45_goodB2G_data = data;
    goodB2G_sink();
}

void CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_45_good()
{
    goodG2B();
    goodB2G();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */
#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_45_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE135_Incorrect_Calculation_Of_Multibyte_String_Length__basic_45_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
