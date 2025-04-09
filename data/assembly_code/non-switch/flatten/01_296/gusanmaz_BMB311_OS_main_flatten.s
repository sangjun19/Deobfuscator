	.file	"gusanmaz_BMB311_OS_main_flatten.c"
	.text
	.globl	_TIG_IZ_OoXR_argv
	.bss
	.align 8
	.type	_TIG_IZ_OoXR_argv, @object
	.size	_TIG_IZ_OoXR_argv, 8
_TIG_IZ_OoXR_argv:
	.zero	8
	.globl	_TIG_IZ_OoXR_envp
	.align 8
	.type	_TIG_IZ_OoXR_envp, @object
	.size	_TIG_IZ_OoXR_envp, 8
_TIG_IZ_OoXR_envp:
	.zero	8
	.globl	_TIG_IZ_OoXR_argc
	.align 4
	.type	_TIG_IZ_OoXR_argc, @object
	.size	_TIG_IZ_OoXR_argc, 4
_TIG_IZ_OoXR_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"r"
.LC1:
	.string	"%s"
.LC2:
	.string	"NULL"
	.text
	.globl	read
	.type	read, @function
read:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movq	%rdi, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$4, -80(%rbp)
.L21:
	cmpq	$11, -80(%rbp)
	ja	.L24
	movq	-80(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L24-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L11:
	movq	$8, -80(%rbp)
	jmp	.L15
.L7:
	movq	-104(%rbp), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -72(%rbp)
	movq	-72(%rbp), %rax
	movq	%rax, -96(%rbp)
	movq	$0, -80(%rbp)
	jmp	.L15
.L13:
	leaq	-64(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$5, -80(%rbp)
	jmp	.L15
.L12:
	cmpq	$0, -88(%rbp)
	jne	.L16
	movq	$9, -80(%rbp)
	jmp	.L15
.L16:
	movq	$1, -80(%rbp)
	jmp	.L15
.L3:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -80(%rbp)
	jmp	.L15
.L6:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$10, -80(%rbp)
	jmp	.L15
.L9:
	movq	-96(%rbp), %rdx
	leaq	-64(%rbp), %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -88(%rbp)
	movq	$3, -80(%rbp)
	jmp	.L15
.L10:
	movl	$0, %eax
	jmp	.L22
.L5:
	movl	$-1, %eax
	jmp	.L22
.L14:
	cmpq	$0, -96(%rbp)
	jne	.L19
	movq	$11, -80(%rbp)
	jmp	.L15
.L19:
	movq	$6, -80(%rbp)
	jmp	.L15
.L8:
	movl	$-1, %eax
	jmp	.L22
.L24:
	nop
.L15:
	jmp	.L21
.L22:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L23
	call	__stack_chk_fail@PLT
.L23:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	read, .-read
	.section	.rodata
.LC3:
	.string	"/proc/%s/comm"
.LC4:
	.string	"parameters failed."
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_OoXR_envp(%rip)
	nop
.L26:
	movq	$0, _TIG_IZ_OoXR_argv(%rip)
	nop
.L27:
	movl	$0, _TIG_IZ_OoXR_argc(%rip)
	nop
	nop
.L28:
.L29:
#APP
# 79 "gusanmaz_BMB311_OS_main.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-OoXR--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_OoXR_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_OoXR_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_OoXR_envp(%rip)
	nop
	movq	$0, -72(%rbp)
.L41:
	cmpq	$5, -72(%rbp)
	ja	.L44
	movq	-72(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L32(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L32(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L32:
	.long	.L36-.L32
	.long	.L44-.L32
	.long	.L35-.L32
	.long	.L34-.L32
	.long	.L33-.L32
	.long	.L31-.L32
	.text
.L33:
	movl	$0, %eax
	jmp	.L42
.L34:
	movl	$1, %eax
	jmp	.L42
.L31:
	movq	-96(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rdx
	leaq	-64(%rbp), %rax
	leaq	.LC3(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	read
	movq	$4, -72(%rbp)
	jmp	.L38
.L36:
	cmpl	$2, -84(%rbp)
	je	.L39
	movq	$2, -72(%rbp)
	jmp	.L38
.L39:
	movq	$5, -72(%rbp)
	jmp	.L38
.L35:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -72(%rbp)
	jmp	.L38
.L44:
	nop
.L38:
	jmp	.L41
.L42:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L43
	call	__stack_chk_fail@PLT
.L43:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
