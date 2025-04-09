	.file	"emacslisp_cpp_getopt_long_flatten.c"
	.text
	.globl	_TIG_IZ_JAKy_argc
	.bss
	.align 4
	.type	_TIG_IZ_JAKy_argc, @object
	.size	_TIG_IZ_JAKy_argc, 4
_TIG_IZ_JAKy_argc:
	.zero	4
	.globl	_TIG_IZ_JAKy_envp
	.align 8
	.type	_TIG_IZ_JAKy_envp, @object
	.size	_TIG_IZ_JAKy_envp, 8
_TIG_IZ_JAKy_envp:
	.zero	8
	.globl	_TIG_IZ_JAKy_argv
	.align 8
	.type	_TIG_IZ_JAKy_argv, @object
	.size	_TIG_IZ_JAKy_argv, 8
_TIG_IZ_JAKy_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"unknown option: %c\n"
.LC1:
	.string	"option needs a value"
.LC2:
	.string	"initialize"
.LC3:
	.string	"file"
.LC4:
	.string	"list"
.LC5:
	.string	"restart"
.LC6:
	.string	"option: %c\n"
.LC7:
	.string	"filename: %s\n"
.LC8:
	.string	"argument: %s\n"
.LC9:
	.string	"if:lr"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$224, %rsp
	movl	%edi, -196(%rbp)
	movq	%rsi, -208(%rbp)
	movq	%rdx, -216(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_JAKy_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_JAKy_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_JAKy_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 122 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-JAKy--0
# 0 "" 2
#NO_APP
	movl	-196(%rbp), %eax
	movl	%eax, _TIG_IZ_JAKy_argc(%rip)
	movq	-208(%rbp), %rax
	movq	%rax, _TIG_IZ_JAKy_argv(%rip)
	movq	-216(%rbp), %rax
	movq	%rax, _TIG_IZ_JAKy_envp(%rip)
	nop
	movq	$7, -184(%rbp)
.L32:
	cmpq	$21, -184(%rbp)
	ja	.L35
	movq	-184(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L20-.L8
	.long	.L35-.L8
	.long	.L19-.L8
	.long	.L35-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L35-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L35-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L12-.L8
	.long	.L35-.L8
	.long	.L11-.L8
	.long	.L35-.L8
	.long	.L10-.L8
	.long	.L35-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	movl	optind(%rip), %eax
	cmpl	%eax, -196(%rbp)
	jle	.L21
	movq	$0, -184(%rbp)
	jmp	.L23
.L21:
	movq	$5, -184(%rbp)
	jmp	.L23
.L18:
	movl	optopt(%rip), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$20, -184(%rbp)
	jmp	.L23
.L12:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -184(%rbp)
	jmp	.L23
.L15:
	cmpl	$-1, -188(%rbp)
	je	.L24
	movq	$2, -184(%rbp)
	jmp	.L23
.L24:
	movq	$18, -184(%rbp)
	jmp	.L23
.L11:
	leaq	.LC2(%rip), %rax
	movq	%rax, -176(%rbp)
	movl	$0, -168(%rbp)
	movq	$0, -160(%rbp)
	movl	$105, -152(%rbp)
	leaq	.LC3(%rip), %rax
	movq	%rax, -144(%rbp)
	movl	$1, -136(%rbp)
	movq	$0, -128(%rbp)
	movl	$102, -120(%rbp)
	leaq	.LC4(%rip), %rax
	movq	%rax, -112(%rbp)
	movl	$0, -104(%rbp)
	movq	$0, -96(%rbp)
	movl	$108, -88(%rbp)
	leaq	.LC5(%rip), %rax
	movq	%rax, -80(%rbp)
	movl	$0, -72(%rbp)
	movq	$0, -64(%rbp)
	movl	$114, -56(%rbp)
	movq	$0, -48(%rbp)
	movl	$0, -40(%rbp)
	movq	$0, -32(%rbp)
	movl	$0, -24(%rbp)
	movq	$20, -184(%rbp)
	jmp	.L23
.L7:
	movl	-188(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$20, -184(%rbp)
	jmp	.L23
.L13:
	movq	optarg(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$20, -184(%rbp)
	jmp	.L23
.L17:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L33
	jmp	.L34
.L14:
	movq	$20, -184(%rbp)
	jmp	.L23
.L20:
	movl	optind(%rip), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-208(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	optind(%rip), %eax
	addl	$1, %eax
	movl	%eax, optind(%rip)
	movq	$18, -184(%rbp)
	jmp	.L23
.L16:
	movq	$16, -184(%rbp)
	jmp	.L23
.L19:
	cmpl	$114, -188(%rbp)
	je	.L27
	cmpl	$114, -188(%rbp)
	jg	.L28
	cmpl	$108, -188(%rbp)
	je	.L27
	cmpl	$108, -188(%rbp)
	jg	.L28
	cmpl	$105, -188(%rbp)
	je	.L27
	cmpl	$105, -188(%rbp)
	jg	.L28
	cmpl	$102, -188(%rbp)
	je	.L29
	cmpl	$102, -188(%rbp)
	jg	.L28
	cmpl	$58, -188(%rbp)
	je	.L30
	cmpl	$63, -188(%rbp)
	jne	.L28
	movq	$4, -184(%rbp)
	jmp	.L31
.L30:
	movq	$14, -184(%rbp)
	jmp	.L31
.L29:
	movq	$11, -184(%rbp)
	jmp	.L31
.L27:
	movq	$21, -184(%rbp)
	jmp	.L31
.L28:
	movq	$10, -184(%rbp)
	nop
.L31:
	jmp	.L23
.L9:
	leaq	-176(%rbp), %rdx
	movq	-208(%rbp), %rsi
	movl	-196(%rbp), %eax
	movl	$0, %r8d
	movq	%rdx, %rcx
	leaq	.LC9(%rip), %rdx
	movl	%eax, %edi
	call	getopt_long@PLT
	movl	%eax, -188(%rbp)
	movq	$8, -184(%rbp)
	jmp	.L23
.L35:
	nop
.L23:
	jmp	.L32
.L34:
	call	__stack_chk_fail@PLT
.L33:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
