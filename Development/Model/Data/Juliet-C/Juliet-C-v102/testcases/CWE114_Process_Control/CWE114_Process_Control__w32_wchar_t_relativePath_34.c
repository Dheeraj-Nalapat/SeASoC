/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE114_Process_Control__w32_wchar_t_relativePath_34.c
Label Definition File: CWE114_Process_Control__w32.label.xml
Template File: sources-sink-34.tmpl.c
*/
/*
 * @description
 * CWE: 114 Process Control
 * BadSource: relativePath Hard code the relative pathname to the library
 * GoodSource: Hard code the full pathname to the library
 * Sinks:
 *    BadSink : Load a dynamic link library
 * Flow Variant: 34 Data flow: use of a union containing two methods of accessing the same data (within the same function)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#include <windows.h>

typedef union
{
    wchar_t * a;
    wchar_t * b;
} CWE114_Process_Control__w32_wchar_t_relativePath_34_union_type;

#ifndef OMITBAD

void CWE114_Process_Control__w32_wchar_t_relativePath_34_bad()
{
    wchar_t * data;
    CWE114_Process_Control__w32_wchar_t_relativePath_34_union_type my_union;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    /* FLAW: Specify just the file name for the library, not the full path */
    wcscpy(data, L"winsrv.dll");
    my_union.a = data;
    {
        wchar_t * data = my_union.b;
#ifdef _WIN32
        {
            HMODULE hModule;
            /* POTENTIAL FLAW: If the path to the library is not specified, an attacker may be able to
             * replace his own file with the intended library */
            hModule = LoadLibraryW(data);
            if (hModule != NULL)
            {
                FreeLibrary(hModule);
                printLine("Library loaded and freed successfully");
            }
            else
            {
                printLine("Unable to load library");
            }
        }
#endif
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    wchar_t * data;
    CWE114_Process_Control__w32_wchar_t_relativePath_34_union_type my_union;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    /* FIX: Specify the full pathname for the library */
    wcscpy(data, L"C:\\Windows\\System32\\winsrv.dll");
    my_union.a = data;
    {
        wchar_t * data = my_union.b;
#ifdef _WIN32
        {
            HMODULE hModule;
            /* POTENTIAL FLAW: If the path to the library is not specified, an attacker may be able to
             * replace his own file with the intended library */
            hModule = LoadLibraryW(data);
            if (hModule != NULL)
            {
                FreeLibrary(hModule);
                printLine("Library loaded and freed successfully");
            }
            else
            {
                printLine("Unable to load library");
            }
        }
#endif
    }
}

void CWE114_Process_Control__w32_wchar_t_relativePath_34_good()
{
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
    CWE114_Process_Control__w32_wchar_t_relativePath_34_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE114_Process_Control__w32_wchar_t_relativePath_34_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
