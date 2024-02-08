/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE23_Relative_Path_Traversal__wchar_t_Environment_open_67b.c
Label Definition File: CWE23_Relative_Path_Traversal.label.xml
Template File: sources-sink-67b.tmpl.c
*/
/*
 * @description
 * CWE: 23 Relative Path Traversal
 * BadSource: Environment Read input from an environment variable
 * GoodSource: File name without a period or slash
 * Sinks: open
 *    BadSink :
 * Flow Variant: 67 Data flow: data passed in a struct from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
#define BASEPATH L"c:\\temp\\"
#else
#define BASEPATH L"/tmp/"
#endif

#define ENV_VARIABLE L"ADD"

#ifdef _WIN32
# define GETENV _wgetenv
#else
# define GETENV wgetenv
#endif

#ifdef _WIN32
# define OPEN _wopen
# define CLOSE _close
#else
# define OPEN wopen
# define CLOSE close
#endif

typedef struct _CWE23_Relative_Path_Traversal__wchar_t_Environment_open_67_struct_type
{
    wchar_t * a;
} CWE23_Relative_Path_Traversal__wchar_t_Environment_open_67_struct_type;

#ifndef OMITBAD

void CWE23_Relative_Path_Traversal__wchar_t_Environment_open_67b_bad_sink(CWE23_Relative_Path_Traversal__wchar_t_Environment_open_67_struct_type my_struct)
{
    wchar_t * data = my_struct.a;
    {
        int fd;
        /* POTENTIAL FLAW: Possibly opening a file without validating the file name or path */
        fd = OPEN(data, O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
        if (fd != -1)
        {
            CLOSE(fd);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE23_Relative_Path_Traversal__wchar_t_Environment_open_67b_goodG2B_sink(CWE23_Relative_Path_Traversal__wchar_t_Environment_open_67_struct_type my_struct)
{
    wchar_t * data = my_struct.a;
    {
        int fd;
        /* POTENTIAL FLAW: Possibly opening a file without validating the file name or path */
        fd = OPEN(data, O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
        if (fd != -1)
        {
            CLOSE(fd);
        }
    }
}

#endif /* OMITGOOD */
