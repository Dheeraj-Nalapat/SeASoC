/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE114_Process_Control__w32_wchar_t_fromConsole_52b.c
Label Definition File: CWE114_Process_Control__w32.label.xml
Template File: sources-sink-52b.tmpl.c
*/
/*
 * @description
 * CWE: 114 Process Control
 * BadSource: fromConsole Read input from the console
 * GoodSource: Hard code the full pathname to the library
 * Sink:
 *    BadSink : Load a dynamic link library
 * Flow Variant: 52 Data flow: data passed as an argument from one function to another to another in three different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#include <windows.h>

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

/* bad function declaration */
void CWE114_Process_Control__w32_wchar_t_fromConsole_52c_bad_sink(wchar_t * data);

void CWE114_Process_Control__w32_wchar_t_fromConsole_52b_bad_sink(wchar_t * data)
{
    CWE114_Process_Control__w32_wchar_t_fromConsole_52c_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declaration */
void CWE114_Process_Control__w32_wchar_t_fromConsole_52c_goodG2B_sink(wchar_t * data);

/* goodG2B uses the GoodSource with the BadSink */
void CWE114_Process_Control__w32_wchar_t_fromConsole_52b_goodG2B_sink(wchar_t * data)
{
    CWE114_Process_Control__w32_wchar_t_fromConsole_52c_goodG2B_sink(data);
}

#endif /* OMITGOOD */
