/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE134_Uncontrolled_Format_String__wchar_t_fromFile_printf_41.c
Label Definition File: CWE134_Uncontrolled_Format_String.label.xml
Template File: sources-sinks-41.tmpl.c
*/
/*
 * @description
 * CWE: 134 Uncontrolled Format String
 * BadSource: fromFile Read input from a file
 * GoodSource: Copy a fixed string into data
 * Sinks: printf
 *    GoodSink: wprintf with "%s" as the first argument and data as the second
 *    BadSink : wprintf with only data as an argument
 * Flow Variant: 41 Data flow: data passed as an argument from one function to another in the same source file
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define FOPEN _wfopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
#endif

#ifndef OMITBAD

static void bad_sink(wchar_t * data)
{
    /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
    wprintf(data);
}

void CWE134_Uncontrolled_Format_String__wchar_t_fromFile_printf_41_bad()
{
    wchar_t * data;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    {
        /* Read input from a file */
        size_t data_len = wcslen(data);
        FILE * pFile;
        /* if there is room in data, attempt to read the input from a file */
        if(100-data_len > 1)
        {
            pFile = FOPEN(L"C:\\temp\\file.txt", L"r");
            if (pFile != NULL)
            {
                fgetws(data+data_len, (int)(100-data_len), pFile);
                fclose(pFile);
            }
        }
    }
    bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B_sink(wchar_t * data)
{
    /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
    wprintf(data);
}

static void goodG2B()
{
    wchar_t * data;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    /* FIX: Use a fixed string that does not contain a format specifier */
    wcscpy(data, L"fixedstringtest");
    goodG2B_sink(data);
}

/* goodB2G uses the BadSource with the GoodSink */
static void goodB2G_sink(wchar_t * data)
{
    /* FIX: Specify the format disallowing a format string vulnerability */
    wprintf(L"%s\n", data);
}

static void goodB2G()
{
    wchar_t * data;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    {
        /* Read input from a file */
        size_t data_len = wcslen(data);
        FILE * pFile;
        /* if there is room in data, attempt to read the input from a file */
        if(100-data_len > 1)
        {
            pFile = FOPEN(L"C:\\temp\\file.txt", L"r");
            if (pFile != NULL)
            {
                fgetws(data+data_len, (int)(100-data_len), pFile);
                fclose(pFile);
            }
        }
    }
    goodB2G_sink(data);
}

void CWE134_Uncontrolled_Format_String__wchar_t_fromFile_printf_41_good()
{
    goodB2G();
    goodG2B();
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
    CWE134_Uncontrolled_Format_String__wchar_t_fromFile_printf_41_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE134_Uncontrolled_Format_String__wchar_t_fromFile_printf_41_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif