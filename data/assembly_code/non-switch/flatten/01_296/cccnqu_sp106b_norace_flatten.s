	.file	"cccnqu_sp106b_norace_flatten.c"
	.text
	.globl	_TIG_IZ_umzo_argv
	.bss
	.align 8
	.type	_TIG_IZ_umzo_argv, @object
	.size	_TIG_IZ_umzo_argv, 8
_TIG_IZ_umzo_argv:
	.zero	8
	.globl	mutex1
	.align 32
	.type	mutex1, @object
	.size	mutex1, 40
mutex1:
	.zero	40
	.globl	counter
	.align 4
	.type	counter, @object
	.size	counter, 4
counter:
	.zero	4
	.globl	_TIG_IZ_umzo_envp
	.align 8
	.type	_TIG_IZ_umzo_envp, @object
	.size	_TIG_IZ_umzo_envp, 8
_TIG_IZ_umzo_envp:
	.zero	8
	.globl	_TIG_IZ_umzo_argc
	.align 4
	.type	_TIG_IZ_umzo_argc, @object
	.size	_TIG_IZ_umzo_argc, 4
_TIG_IZ_umzo_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"counter=%d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, counter(%rip)
	nop
.L2:
	movl	$0, mutex1(%rip)
	movl	$0, 4+mutex1(%rip)
	movl	$0, 8+mutex1(%rip)
	movl	$0, 12+mutex1(%rip)
	movl	$0, 16+mutex1(%rip)
	movw	$0, 20+mutex1(%rip)
	movw	$0, 22+mutex1(%rip)
	movq	$0, 24+mutex1(%rip)
	movq	$0, 32+mutex1(%rip)
	nop
.L3:
	movq	$0, _TIG_IZ_umzo_envp(%rip)
	nop
.L4:
	movq	$0, _TIG_IZ_umzo_argv(%rip)
	nop
.L5:
	movl	$0, _TIG_IZ_umzo_argc(%rip)
	nop
	nop
.L6:
.L7:
#APP
# 123 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-umzo--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_umzo_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_umzo_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_umzo_envp(%rip)
	nop
	movq	$0, -16(%rbp)
.L13:
	cmpq	$2, -16(%rbp)
	je	.L8
	cmpq	$2, -16(%rbp)
	ja	.L16
	cmpq	$0, -16(%rbp)
	je	.L10
	cmpq	$1, -16(%rbp)
	jne	.L16
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L14
	jmp	.L15
.L10:
	movq	$2, -16(%rbp)
	jmp	.L12
.L8:
	leaq	-32(%rbp), %rax
	movl	$0, %ecx
	leaq	inc(%rip), %rdx
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_create@PLT
	leaq	-24(%rbp), %rax
	movl	$0, %ecx
	leaq	dec(%rip), %rdx
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_create@PLT
	movq	-32(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_join@PLT
	movq	-24(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_join@PLT
	movl	counter(%rip), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L12
.L16:
	nop
.L12:
	jmp	.L13
.L15:
	call	__stack_chk_fail@PLT
.L14:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.globl	inc
	.type	inc, @function
inc:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$3, -8(%rbp)
.L27:
	cmpq	$6, -8(%rbp)
	je	.L18
	cmpq	$6, -8(%rbp)
	ja	.L29
	cmpq	$3, -8(%rbp)
	je	.L20
	cmpq	$3, -8(%rbp)
	ja	.L29
	cmpq	$1, -8(%rbp)
	je	.L21
	cmpq	$2, -8(%rbp)
	je	.L22
	jmp	.L29
.L21:
	cmpl	$99999, -12(%rbp)
	jg	.L23
	movq	$6, -8(%rbp)
	jmp	.L25
.L23:
	movq	$2, -8(%rbp)
	jmp	.L25
.L20:
	movl	$0, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L25
.L18:
	leaq	mutex1(%rip), %rax
	movq	%rax, %rdi
	call	pthread_mutex_lock@PLT
	movl	counter(%rip), %eax
	addl	$1, %eax
	movl	%eax, counter(%rip)
	leaq	mutex1(%rip), %rax
	movq	%rax, %rdi
	call	pthread_mutex_unlock@PLT
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L25
.L22:
	movl	$0, %eax
	jmp	.L28
.L29:
	nop
.L25:
	jmp	.L27
.L28:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	inc, .-inc
	.globl	dec
	.type	dec, @function
dec:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L40:
	cmpq	$6, -8(%rbp)
	je	.L31
	cmpq	$6, -8(%rbp)
	ja	.L42
	cmpq	$4, -8(%rbp)
	je	.L33
	cmpq	$4, -8(%rbp)
	ja	.L42
	cmpq	$0, -8(%rbp)
	je	.L34
	cmpq	$3, -8(%rbp)
	je	.L35
	jmp	.L42
.L33:
	leaq	mutex1(%rip), %rax
	movq	%rax, %rdi
	call	pthread_mutex_lock@PLT
	movl	counter(%rip), %eax
	subl	$1, %eax
	movl	%eax, counter(%rip)
	leaq	mutex1(%rip), %rax
	movq	%rax, %rdi
	call	pthread_mutex_unlock@PLT
	addl	$1, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L36
.L35:
	cmpl	$99999, -12(%rbp)
	jg	.L37
	movq	$4, -8(%rbp)
	jmp	.L36
.L37:
	movq	$6, -8(%rbp)
	jmp	.L36
.L31:
	movl	$0, %eax
	jmp	.L41
.L34:
	movl	$0, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L36
.L42:
	nop
.L36:
	jmp	.L40
.L41:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	dec, .-dec
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
