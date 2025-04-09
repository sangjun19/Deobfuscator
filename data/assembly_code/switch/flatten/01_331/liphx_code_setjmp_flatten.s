	.file	"liphx_code_setjmp_flatten.c"
	.text
	.globl	_TIG_IZ_aZ8C_envp
	.bss
	.align 8
	.type	_TIG_IZ_aZ8C_envp, @object
	.size	_TIG_IZ_aZ8C_envp, 8
_TIG_IZ_aZ8C_envp:
	.zero	8
	.globl	env
	.align 32
	.type	env, @object
	.size	env, 200
env:
	.zero	200
	.globl	_TIG_IZ_aZ8C_argv
	.align 8
	.type	_TIG_IZ_aZ8C_argv, @object
	.size	_TIG_IZ_aZ8C_argv, 8
_TIG_IZ_aZ8C_argv:
	.zero	8
	.globl	_TIG_IZ_aZ8C_argc
	.align 4
	.type	_TIG_IZ_aZ8C_argc, @object
	.size	_TIG_IZ_aZ8C_argc, 4
_TIG_IZ_aZ8C_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"func1 called"
	.text
	.type	func1, @function
func1:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$1, -8(%rbp)
.L13:
	cmpq	$4, -8(%rbp)
	ja	.L14
	movq	-8(%rbp), %rax
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
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L15-.L4
	.long	.L3-.L4
	.text
.L3:
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	func2
	movq	$3, -8(%rbp)
	jmp	.L9
.L7:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -8(%rbp)
	jmp	.L9
.L8:
	movl	$1, %esi
	leaq	env(%rip), %rax
	movq	%rax, %rdi
	call	longjmp@PLT
.L6:
	cmpl	$0, -20(%rbp)
	jns	.L11
	movq	$0, -8(%rbp)
	jmp	.L9
.L11:
	movq	$4, -8(%rbp)
	jmp	.L9
.L14:
	nop
.L9:
	jmp	.L13
.L15:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	func1, .-func1
	.section	.rodata
.LC1:
	.string	"func2 called"
	.text
	.type	func2, @function
func2:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$2, -8(%rbp)
.L25:
	cmpq	$3, -8(%rbp)
	je	.L26
	cmpq	$3, -8(%rbp)
	ja	.L27
	cmpq	$2, -8(%rbp)
	je	.L19
	cmpq	$2, -8(%rbp)
	ja	.L27
	cmpq	$0, -8(%rbp)
	je	.L20
	cmpq	$1, -8(%rbp)
	jne	.L27
	movl	$2, %esi
	leaq	env(%rip), %rax
	movq	%rax, %rdi
	call	longjmp@PLT
.L20:
	movl	-20(%rbp), %eax
	andl	$1, %eax
	testl	%eax, %eax
	jne	.L22
	movq	$1, -8(%rbp)
	jmp	.L24
.L22:
	movq	$3, -8(%rbp)
	jmp	.L24
.L19:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L24
.L27:
	nop
.L24:
	jmp	.L25
.L26:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	func2, .-func2
	.section	.rodata
.LC2:
	.string	"func2 error\n"
.LC3:
	.string	"func1 error\n"
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
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, env(%rip)
	movq	$0, 8+env(%rip)
	movq	$0, 16+env(%rip)
	movq	$0, 24+env(%rip)
	movq	$0, 32+env(%rip)
	movq	$0, 40+env(%rip)
	movq	$0, 48+env(%rip)
	movq	$0, 56+env(%rip)
	movl	$0, 64+env(%rip)
	movq	$0, 72+env(%rip)
	movq	$0, 80+env(%rip)
	movq	$0, 88+env(%rip)
	movq	$0, 96+env(%rip)
	movq	$0, 104+env(%rip)
	movq	$0, 112+env(%rip)
	movq	$0, 120+env(%rip)
	movq	$0, 128+env(%rip)
	movq	$0, 136+env(%rip)
	movq	$0, 144+env(%rip)
	movq	$0, 152+env(%rip)
	movq	$0, 160+env(%rip)
	movq	$0, 168+env(%rip)
	movq	$0, 176+env(%rip)
	movq	$0, 184+env(%rip)
	movq	$0, 192+env(%rip)
	nop
.L29:
	movq	$0, _TIG_IZ_aZ8C_envp(%rip)
	nop
.L30:
	movq	$0, _TIG_IZ_aZ8C_argv(%rip)
	nop
.L31:
	movl	$0, _TIG_IZ_aZ8C_argc(%rip)
	nop
	nop
.L32:
.L33:
#APP
# 152 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-aZ8C--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_aZ8C_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_aZ8C_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_aZ8C_envp(%rip)
	nop
	movq	$8, -8(%rbp)
.L49:
	cmpq	$10, -8(%rbp)
	ja	.L51
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L36(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L36(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L36:
	.long	.L41-.L36
	.long	.L51-.L36
	.long	.L40-.L36
	.long	.L51-.L36
	.long	.L51-.L36
	.long	.L51-.L36
	.long	.L39-.L36
	.long	.L51-.L36
	.long	.L50-.L36
	.long	.L37-.L36
	.long	.L35-.L36
	.text
.L50:
	leaq	env(%rip), %rax
	movq	%rax, %rdi
	call	_setjmp@PLT
	endbr64
	movl	%eax, -12(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L43
.L37:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$12, %edx
	movl	$1, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$2, %edi
	call	exit@PLT
.L39:
	movl	$2, %edi
	call	func1
	movl	$0, %edi
	call	exit@PLT
.L35:
	cmpl	$2, -12(%rbp)
	je	.L44
	cmpl	$2, -12(%rbp)
	jg	.L45
	cmpl	$0, -12(%rbp)
	je	.L46
	cmpl	$1, -12(%rbp)
	je	.L47
	jmp	.L45
.L44:
	movq	$9, -8(%rbp)
	jmp	.L48
.L47:
	movq	$0, -8(%rbp)
	jmp	.L48
.L46:
	movq	$6, -8(%rbp)
	jmp	.L48
.L45:
	movq	$2, -8(%rbp)
	nop
.L48:
	jmp	.L43
.L41:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$12, %edx
	movl	$1, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L40:
	movl	$-1, %edi
	call	exit@PLT
.L51:
	nop
.L43:
	jmp	.L49
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
